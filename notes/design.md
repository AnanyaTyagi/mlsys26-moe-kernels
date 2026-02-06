# Design notes (MoE Track)

Goals:
- Implement and optimize FP8 fused MoE kernels for the MLSys 2026 NVIDIA Track.

Plan:
- Start from baseline / starter kit structure
- Identify bottlenecks (routing, memory traffic, load balance)
- Implement v1 optimized kernel variant
