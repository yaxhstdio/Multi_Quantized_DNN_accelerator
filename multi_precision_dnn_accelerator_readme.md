# Multi-Precision DNN Hardware Accelerator (INT4 / INT8 / INT16)

A reconfigurable, low-latency Deep Neural Network (DNN) Multiply-Accumulate (MAC) accelerator IP core designed in Verilog HDL. Built for edge AI inference, the architecture dynamically switches between **INT4, INT8, and INT16** precision modes at runtime, balancing throughput, memory bandwidth, and model accuracy.

The design is functionally verified via simulation (Icarus Verilog + Python golden model) and synthesized on a **Xilinx Artix-7 FPGA (`xc7a200tfbg676-2`)** using AMD Vivado v2026.1.

---

## 1. Architecture & Module Hierarchy

The accelerator uses a lean, direct-stream datapath that eliminates intermediate parameter and activation buffers, minimizing cycle latency and LUT overhead.

```text
accelerator_top.v                 # Top-level physical chip wrapper
├── parameter_bram.v              # 64x64-bit on-chip weight memory (weights.hex)
├── mac_control_unit.v            # 4-stage FSM (IDLE, LOAD, COMPUTE, DONE)
├── mac_array_requantized.v       # 4-PE parallel compute array (genvar instantiated)
│   ├── mode_decoder.v            # Dynamic precision control (INT4 / INT8 / INT16)
│   ├── pe_mac.v [0..3]           # Processing Elements mapped to DSP48E1 slices
│   └── requantize_var.v [0..3]   # Dynamic bit-shift requantizer + parameterized ReLU
└── sync_fifo.v                   # Output activation buffer
```

### Key Design Highlights
* **Dynamic Multi-Precision Compute:** Controlled via a 2-bit `mode_in` signal, allowing the same 4-PE array to process INT4, INT8, or INT16 workloads without resynthesizing the hardware.
* **4-Stage Low-Latency FSM:** By wiring `parameter_bram` directly into `mac_array_requantized`, the controller eliminates intermediate buffer-shuffling states, reducing control-path LUTs and total inference latency.
* **Synthesis-Safe Full-Array Mapping:** To prevent Vivado's dead-logic elimination from pruning parallel PEs when routing to a single 16-bit FIFO, `accelerator_top.v` implements an XOR reduction tree across all four PE outputs. This forces true physical mapping of all 4 DSP48E1 slices for accurate PPA evaluation.
* **Hardware-Agnostic Coprocessor Model:** Designed as a standalone compute IP core (leaving final `argmax` classification to the host CPU), mirroring modern industry NPU/coprocessor architectures.

---

## 2. PPA (Power, Performance, Area) Summary

* **Target Device:** Xilinx Artix-7 (`xc7a200tfbg676-2`, Speed Grade `-2`)
* **Synthesis Tool:** AMD Vivado v2026.1

### Area (Resource Utilization)
| Resource | Used | Available | Utilization (%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Slice LUTs** | 752 | 134,600 | 0.56% | FSM, mode decoder, 4x requantizers, ReLU, and distributed weight ROM |
| **Slice Registers (FFs)** | 295 | 269,200 | 0.11% | Pipeline registers, accumulators, and FIFO pointers |
| **DSP48E1 Slices** | 4 | 740 | 0.54% | 1 dedicated DSP slice per Processing Element |
| **Block RAM (BRAM)** | 0 | 365 | 0.00% | 64-word weight memory automatically inferred as Distributed RAM |
| **Bonded IOBs** | 47 | 400 | 11.75% | External parallel interface pins |

### Performance (Static Timing Analysis)
* **Target Clock Period:** $10.000\text{ ns}$ ($100\text{ MHz}$)
* **Worst Negative Slack (WNS):** $+3.139\text{ ns}$ (Timing Met)
* **Actual Critical Path Delay:** $6.861\text{ ns}$
* **Maximum Clock Frequency ($F_{max}$):** **$\approx 145.75\text{ MHz}$**

### Power Estimation (Post-Synthesis)
| Metric | Power (W) | Share (%) |
| :--- | :---: | :---: |
| **Total On-Chip Power** | **0.165 W** | **100%** |
| Device Static Power | 0.131 W | 79% |
| Dynamic Power | 0.034 W | 21% |
| - I/O | 0.011 W | 33% of Dynamic |
| - Logic (LUTs) | 0.008 W | 24% of Dynamic |
| - Signals | 0.008 W | 23% of Dynamic |
| - DSPs | 0.004 W | 12% of Dynamic |
| - Clocks | 0.003 W | 8% of Dynamic |

*Junction Temperature: $25.3^\circ\text{C}$ | Thermal Margin: $59.7^\circ\text{C}$*

---

## 3. Architectural Design & Trade-Offs

Compared to standalone peripheral-heavy SoC designs (e.g., UART-driven accelerators with hardcoded Argmax and 7-segment display drivers), this architecture prioritizes core compute efficiency and clean physical design:

* **Latency vs. Fan-Out Scalability:** Direct BRAM-to-MAC wiring eliminates intermediate parameter buffers and reduces the FSM to 4 states, minimizing cycle latency. Scaling to massive arrays (e.g., 256+ PEs) would require reintroducing pipeline buffers to manage physical routing fan-out.
* **Flexibility vs. Autonomy:** Keeping `argmax` classification in software makes the IP workload-agnostic (usable for vision, audio, or general matrix math) at the cost of requiring a host processor to drive control flow and read the output FIFO.
* **Bus Speed vs. Plug-and-Play I/O:** Omitting UART peripherals avoids serial bandwidth bottlenecks and keeps PPA metrics unpolluted, while preparing the core for high-speed AXI memory-mapped integration.

---

## 4. Precision & Quantization Rationale (FP32 vs. Integer)

Training neural networks relies on **FP32** to capture tiny gradient updates during backpropagation, but FP32 arithmetic incurs massive LUT, timing, and power penalties on edge FPGAs. For **edge inference**, Post-Training Quantization (PTQ) maps locked FP32 weights into fixed integer formats that map directly to DSP48E1 slices:

* **INT16:** Near-identical accuracy to FP32 with substantial hardware savings.
* **INT8:** Edge AI industry standard; reduces memory bandwidth by $4\times$ vs. FP32 with typically $<1\%$ accuracy loss.
* **INT4:** Extreme compression ($8\times$ bandwidth reduction); ideal for memory-constrained workloads, typically paired with Quantization-Aware Training (QAT) to preserve accuracy.

---

## 5. Simulation & Verification

Functional verification is performed using **Icarus Verilog** and **GTKWave**, validated against a Python golden reference model (`golden_model.py`).

### Prerequisites
* `iverilog` and `vvp`
* `gtkwave` (optional, for waveform inspection)
* Python 3 (for golden model verification)

### Running the Top-Level Testbench
Ensure `weights.hex` is present in the project root directory, then compile and run:

```bash
iverilog -o sim_accel_top.vvp \
    mode_decoder.v \
    pe_mac.v \
    requantize_var.v \
    mac_array_requantized.v \
    sync_fifo.v \
    parameter_bram.v \
    mac_control_unit.v \
    accelerator_top.v \
    tb_accelerator_top.v

vvp sim_accel_top.vvp
```

### Expected Output
```text
========================================
   TOP WRAPPER SIMULATION RESULTS
========================================
FIFO Empty : 1
FIFO Data  : 16
========================================
```

To inspect internal FSM states, BRAM reads, and MAC pipeline waveforms:
```bash
gtkwave tb_accelerator_top.vcd
```

---

## 6. Roadmap: PYNQ-Z2 SoC & Live MNIST Inference

To evolve this compute IP into a live handwritten digit classifier (28x28 MNIST images = 784 features) on a **Xilinx PYNQ-Z2 (`xc7z020clg400-1`)**:

1. **Model Training & PTQ:** Train an MLP/CNN on MNIST in PyTorch/TensorFlow, apply Post-Training Quantization to INT8/INT4, and export weights.
2. **AXI Wrapper & DMA:** Wrap `accelerator_top.v` with an **AXI4-Lite** control interface (for `start`, `mode_in`, `layer_length`) and an **AXI-Stream (AXIS)** data interface connected to an **AXI DMA** to stream large weight matrices from the board's 512 MB DDR RAM.
3. **Vivado Block Design:** Integrate the Zynq-7000 Processing System (ARM Cortex-A9), AXI DMA, and custom Accelerator IP, then generate the `.bit` and `.hwh` files.
4. **PYNQ Python Driver:** Load the overlay in a Jupyter Notebook on the PYNQ-Z2, stream quantized image tensors via DMA, read output activations from the FIFO, and run `numpy.argmax()` in software for final digit classification.