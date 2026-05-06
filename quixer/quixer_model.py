# Adapted from Quixer (https://github.com/Quantinuum/Quixer),
# Copyright (c) Quantinuum Ltd., licensed under the Apache License 2.0.
# Modifications (2026): replaced token/vocabulary inputs with image-patch
# embeddings and a classification head for vision tasks.
# See LICENSE for the full Apache 2.0 text.

import itertools
from math import log2
from typing import Any

import torch
from torch.types import Device
import torchquantum as tq
from torchquantum import GeneralEncoder, QuantumDevice


def ansatz_14_torchquantum_specification(
    n_qubits: int, layers: int = 1
) -> list[dict[str, Any]]:
    """
    Produces a TorchQuantum specification for the parameterized quantum circuit "ansatz 14" from https://arxiv.org/abs/1905.10876

    Args:
      n_qubits:
        Number of qubits in the circuit
      layers:
        Number of circuit layers
    Returns:
      A TorchQuantum specification for the parameterized quantum circuit "ansatz 14"
    """
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        # First layer of R_Y rotations
        enc.extend(
            [
                {"input_idx": [next(counter)], "func": "ry", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        # First layer of Controlled R_X rotations
        enc.extend(
            [
                {
                    "input_idx": [next(counter)],
                    "func": "crx",
                    "wires": [i, (i + 1) % n_qubits],
                }
                for i in range(n_qubits - 1, -1, -1)
            ]
        )
        # Second layer of R_Y rotations
        enc.extend(
            [
                {"input_idx": [next(counter)], "func": "ry", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        # Second layer of Controlled R_X roations
        enc.extend(
            [
                {
                    "input_idx": [next(counter)],
                    "func": "crx",
                    "wires": [i, (i - 1) % n_qubits],
                }
                for i in [n_qubits - 1] + list(range(n_qubits - 1))
            ]
        )
    return enc


def apply_qsvt_and_lcu(
    initial_states: torch.Tensor,
    pqc_parameters: torch.Tensor,
    parameterized_quantum_circuit: GeneralEncoder,
    torchquantum_device: QuantumDevice,
    n_qubits: int,
    lcu_coefficients: torch.Tensor,
    qsvt_polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Applies the Linear Combination of Unitaries and the Quantum Singular Value Decomposition polynomial
    to a set of initial states. The unitaries are given by a parameterized quantum circuit
    `parameterized_quantum_circuit` with angles `pqc_parameters`. Note that the QSVT and
    the LCU are applied via classical simulation and the full quantum circuit implementing
    these constructs are not reproduced here.

    Args:
      initial_states: Batch of initial quantum states.
      pqc_parameters: Batch of parameters for the parameterized quantum circuits encoding the token embeddings.
      parameterized_quantum_circuit: TorchQuantum specification of the parameterized quantum circuit.
      torchquantum_device: TorchQuantum object representing quantum device.
      n_qubits: Number of qubits for quantum states corresponding to the token embeddings.
      lcu_coefficients: Coefficients of the linear combination of unitaries.
      qsvt_polynomial_coefficients: Coefficients of the polynomial applied using the QSVT.

    """
    # Variable tracking the quantum state as the terms of the polynomial in the QSVT are applied
    # Starts by applying constant term in the polynomial
    accumulated_state = qsvt_polynomial_coefficients[0] * initial_states

    # Intermediate variable storing quantum state as powers of the LCU are applied to it
    # during construction of the QSVT polynomial. In each iteration of the loop below
    # the exponent of the power is increased by one and added to the polynomial
    monomial_state = initial_states
    for c in qsvt_polynomial_coefficients[1:]:
        monomial_state = apply_linear_combination_of_unitaries(
            monomial_state,
            pqc_parameters,
            parameterized_quantum_circuit,
            torchquantum_device,
            n_qubits,
            lcu_coefficients,
        )
        accumulated_state = accumulated_state + c * monomial_state

    return accumulated_state / torch.linalg.vector_norm(
        qsvt_polynomial_coefficients, ord=1
    )


def apply_linear_combination_of_unitaries(
    initial_states: torch.Tensor,
    pqc_parameters: torch.Tensor,
    parameterized_quantum_circuit: GeneralEncoder,
    torchquantum_device: QuantumDevice,
    n_qubits: int,
    lcu_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a linear combination of unitaries to a set of initial states. The unitaries
    are obtained from a parameterized quantum circuit with the sets of parameters `pqc_parameters`.
    Note that the LCU is applied via classical simulation and the full quantum circuit implementing
    this constructs is not reproduced here.

    Args:
      initial_states: Batch of initial quantum states.
      pqc_parameters: Batch of parameters for the parameterized quantum circuits encoding the token embeddings.
      parameterized_quantum_circuit: TorchQuantum specification of the parameterized quantum circuit.
      torchquantum_device: TorchQuantum object representing quantum device.
      n_qubits: Number of qubits for quantum states corresponding to the token embeddings.
      lcu_coefficients: Coefficients of the linear combination of unitaries.

    """
    # Initial state repeated along the batch dimension
    repeated_initial_state = initial_states.repeat(1, pqc_parameters.shape[1]).view(
        -1, 2**n_qubits
    )

    # Set internal TorchQuantum device state to initial states
    torchquantum_device.set_states(repeated_initial_state)

    # Apply circuit to initial states
    parameterized_quantum_circuit(
        torchquantum_device, pqc_parameters.view(-1, pqc_parameters.shape[-1])
    )

    # Extract states from TorchQuantum device
    states = torchquantum_device.get_states_1d().view(
        *pqc_parameters.shape[:2], 2**n_qubits
    )

    # Sum evolved states weighed by LCU coefficients
    lcu_applied_to_state = torch.einsum("bwi,bw->bi", states, lcu_coefficients)

    return lcu_applied_to_state


class Quixer(torch.nn.Module):
    def __init__(
        self,
        n_qubits: int,
        patch_size: int,
        qsvt_polynomial_degree: int,
        n_ansatz_layers: int,
        num_classes: int,
        embedding_dimension: int,
        dropout: float,
        batch_size: int,
        device: Device,
    ):
        """
        Args:
          n_qubits:
            Number of qubits that the unitary embeddings of the patches are defined on.
          n_patches:
            Number of image patches to process.
          qsvt_polynomial_degree:
            Degree of quantum singular value transformation polynomial (e.g. 2 for a quadratic polynomial, 3 for a cubic polynomial, etc).
          n_ansatz_layers:
            Number of layers of the parameterized quantum circuit ansatz used in the unitary embeddings of the patches.
          num_classes:
            Number of output classes for classification.
          embedding_dimension:
            Size of classical embedding vector for each patch.
          dropout:
            Dropout rate (see https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html).
          batch_size:
            Size of each batch.
          device:
            Torch device the model will be classically simulated on.
        """

        super().__init__()

        self.n_qubits = n_qubits

        assert qsvt_polynomial_degree > 0
        self.degree = qsvt_polynomial_degree
        self.device = device

        # Number of parameters in the parameterized quantum circuit ansatz
        self.n_pqc_parameters = 4 * n_qubits * n_ansatz_layers

        self.embedding_dimension = embedding_dimension
        
        # Patch size and image size
        self.patch_size = patch_size
        self.img_size = 224
        assert self.img_size % self.patch_size == 0
        self.num_patches_total = (self.img_size // self.patch_size) ** 2

        self.n_patches = self.num_patches_total
        self.n_ctrl_qubits = int(log2(self.n_patches))
        
        # Patch embedding layer - converts flattened patches to embeddings
        self.patch_embedding = torch.nn.Linear(
            self.patch_size * self.patch_size, self.embedding_dimension
        )

        self.positional_features = torch.nn.Parameter(
            torch.randn(self.num_patches_total, self.embedding_dimension)
        )

        # Xavier uniform initialisation helps training
        torch.nn.init.xavier_uniform_(self.patch_embedding.weight)

        # Linear layer converting the embeddings to angles for the parameterized circuit
        self.embedding_to_angles = torch.nn.Linear(
            in_features=self.embedding_dimension, out_features=self.n_pqc_parameters
        )

        self.dropout = torch.nn.Dropout(dropout)

        # TorchQuantum representation of quantum device
        # Internally holds the statevector to be manipulated
        self.torchquantum_device = tq.QuantumDevice(
            n_wires=self.n_qubits, bsz=batch_size
        )

        # TorchQuantum representation of "ansatz 14" parameterized quantum circuit
        # uses `n_ansatz_layers` layers
        self.token_parameterized_quantum_circuit = tq.GeneralEncoder(
            ansatz_14_torchquantum_specification(n_qubits, n_ansatz_layers)
        )

        self.n_polynomial_coefficients = self.degree + 1

        # Polynomial coefficients for the Quantum Singular Value Transformation
        # e.g. if the QSVT applies the polynomial c_2 x^2 + c_1 x + c_0 then
        # these are the coefficients [c_2, c_1, c_0]
        self.qsvt_polynomial_coefficients = torch.nn.Parameter(
            torch.rand(self.n_polynomial_coefficients)
        )

        # Coefficients for the Linear Combination of Unitaries
        # i.e. if the LCU takes the form \sum_i b_i U_i then these
        # are the coefficients [b_0, ..., b_n]
        self.lcu_coefficients = torch.nn.Parameter(
            torch.rand(self.n_patches, dtype=torch.complex64)
        )

        # TorchQuantum representation of "ansatz 14" parameterized quantum circuit
        # uses 1 layer
        self.quantum_feedforward = tq.GeneralEncoder(
            ansatz_14_torchquantum_specification(n_qubits)
        )

        self.quantum_feedforward_parameters = torch.nn.Parameter(
            torch.rand(self.n_pqc_parameters)
        )

        self.measure_all_x_y_z = tq.MeasureMultipleTimes(
            [
                # X mesurements on all qubits
                {"wires": range(n_qubits), "observables": ["x"] * n_qubits},
                # Y measurements on all qubits
                {"wires": range(n_qubits), "observables": ["y"] * n_qubits},
                # Z measurements on all qubits
                {"wires": range(n_qubits), "observables": ["z"] * n_qubits},
            ]
        )

        self.nr_of_measurements = 3 * n_qubits

        # Output classification layer
        # Produces logits for the output distribution over classes
        self.output_feedforward = torch.nn.Sequential(
            torch.nn.Linear(self.nr_of_measurements, self.embedding_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dimension, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract patches from images
        # Unfold creates patches: [batch_size, 1, num_patches_h, num_patches_w, patch_h, patch_w]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape to [batch_size, num_patches, patch_size * patch_size]
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)

        # Safety check: number of patches matches expected number
        assert patches.shape[1] == self.n_patches, (
            f"Expected {self.n_patches} patches, got {patches.shape[1]}"
        )
                       
        # Tensor with shape [batch_size, n_patches]
        lcu_coefficients = self.lcu_coefficients.repeat(batch_size, 1)
        lcu_coefficients = torch.nn.functional.normalize(lcu_coefficients, p=1)

        # Get patch embeddings
        pos = self.positional_features              # [n_patches, embedding_dim]
        x = self.patch_embedding(patches) + pos

        # Get PQC angles corresponding to each patch
        # Tensor with shape [batch_size, n_patches, n_pqc_parameters]
        pqc_angles = self.embedding_to_angles(self.dropout(x))

        # Initialise |0> state across all batches
        # Tensor with shape [batch_size, 2**n_qubits]
        initial_states = torch.zeros(
            batch_size, 2**self.n_qubits, dtype=torch.complex64, device=self.device
        )
        initial_states[:, 0] = 1.0

        # Quantum state resulting from the application of the QSVT and the LCU
        # Tensor with shape [batch_size, 2**n_qubits]
        qsvt_lcu_state = apply_qsvt_and_lcu(
            initial_states,
            pqc_angles,
            self.token_parameterized_quantum_circuit,
            self.torchquantum_device,
            self.n_qubits,
            lcu_coefficients,
            self.qsvt_polynomial_coefficients,
        )

        # Load state after QSVT+LCU application into TorchQuantum device
        self.torchquantum_device.set_states(
            torch.nn.functional.normalize(qsvt_lcu_state, dim=-1)
        )

        # Apply a PQC at the end with a separate set of trainable parameters
        self.quantum_feedforward(
            self.torchquantum_device,
            self.quantum_feedforward_parameters.repeat(1, batch_size),
        )

        # Measure expectation values
        expectation_values = self.measure_all_x_y_z(self.torchquantum_device)
        expectation_values = (
            expectation_values.reshape(3, batch_size, self.n_qubits)
            .moveaxis(0, 1)
            .reshape(batch_size, -1)
        )

        output_logits = self.output_feedforward(expectation_values)

        # Postselection probabilities
        # Tensor with shape [batch_size,]
        final_probabilities = torch.linalg.vector_norm(qsvt_lcu_state, dim=-1)

        return output_logits, torch.mean(final_probabilities)
