# IoT Security – key ideas

A Physical Unclonable Function (PUF) uses tiny manufacturing variations in a chip to produce a unique, device-specific response to a challenge. Because the variations cannot be copied, a PUF acts like a hardware fingerprint and can derive keys without storing them in memory.

A Trusted Platform Module (TPM) is a dedicated security chip that stores cryptographic keys, measures the boot process and supports remote attestation. Keys inside the TPM never leave the chip in plain form.

ASCON is a family of lightweight authenticated encryption algorithms selected by NIST for constrained devices. It provides confidentiality and integrity with a small hardware and energy footprint.

Privacy by Design means privacy is built into a system from the start: data minimisation, purpose limitation and secure defaults instead of adding privacy features later.
