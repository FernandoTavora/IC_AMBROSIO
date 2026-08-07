  IC_AMBROSIO

Light sheets (LSs) based on superpositions of Bessel beams were recently proposed
by our group, in collaboration with groups in Brazil and abroad, as a new technique in
optical holography of continuous depth, without loss of axial resolution, with experimental results indicating the feasibility of their incorporation in applications that demand
high field structuring at milli- and centimeter scales, such as in optical tweezers systems,
2D and 3D holography, optical imaging, virtual and/or augmented reality, volumetric
displays, among others. Originally constructed from the discrete superposition of Bessel
beams of the same frequency, such solutions of the wave equation can be extended to
the micrometric scale by considering, for example, continuous superpositions. The main
objective of this project is to theoretically and computationally realize these continuously
superposed LSs, allowing a high degree of freedom in the choice of two-dimensional patterns of field intensity at the micrometric scale, with prediction of optical scattering and
gradient forces on dipolar dielectric particles. It is expected, with this project, as main
results: (i) to obtain a theoretical formalism that easily allows the design of continuous
LSs for the various applications listed here, and (ii) to develop an initial code to, in the
future, be able to predict, for example, the behavior of dipolar particles in holographic
optical tweezers systems, when under the incidence of continuous LSs. It is expected
that from this project there will eventually be the preparation of at least one paper for
a high-prestige international conference and a possible article for an international journal
in the field.

BEPE - Generalization of the microstructured light sheet technique based on continuous superpositions of Bessel modes for volumetric structuring

Continuous Vector Frozen Waves (CVFW) for 3D Holography and Optical Trapping

Welcome to the official repository for the simulation of Continuous Vector Frozen Waves (CVFWs) at the micrometer scale. This repository fulfills the written production of code documentation, making these algorithms available to the Applied Electromagnetics Group (AEG) and the broader scientific community via GitHub. 

The primary goal of this project is to provide a stable and efficient codebase for generating 3D field patterns on a micrometer scale, tailored for advanced optical trapping, micromanipulation, and volumetric displays. By leveraging the exact analytical solutions of Maxwell's equations via continuous spectral integration (Mackinnon-type waves), these scripts simulate non-paraxial, diffraction-resistant electromagnetic fields with unprecedented morphological control.

Scientific Overview

Traditional optical trapping relies on highly focused Gaussian beams. However, complex 3D micromanipulation requires non-diffracting wave fields, such as Bessel beams, capable of maintaining their transverse profile over long propagation distances. 

This repository implements Longitudinally Structured Light Sheets (LSLSs). By superimposing continuous spectra of Bessel modes, we bypass the limitations of discrete Fourier truncation (Gibbs phenomenon and evanescent waves), achieving high-fidelity 1D, 2D, and 3D intensity profiles with a 1:1 aspect ratio. Furthermore, the codes evaluate the full vector field ( \mathbf{E} = E_x\hat{x} + E_z\hat{z} ), ensuring that the strictly Maxwellian longitudinal component ( E_z ), which emerges from Gauss's Law ( \nabla \cdot \mathbf{E} = 0 ), is accurately mapped for Rayleigh regime force calculations.

Repository Structure: The Four Core Simulators

The repository is divided into four progressively complex Python scripts. Each script takes a morphological target  F(x,z)  (inputted as standard `.png` images) and synthesizes the exact 3D electromagnetic field required to sculpt that light structure in free space.

1. `cvfw_ideal_homogeneous.py` (Code 1)
      Purpose:   Simulates a homogeneous 3D volume (e.g., extruding a single "F=MA" image across multiple parallel planes).
      Physics:   Operates in the Ideal Regime. It assumes perfect electromagnetic isolation between adjacent light sheets. Crosstalk is mathematically disabled ( y_p = y_{plane} ), providing the "ground truth" of the maximum theoretical contrast of the Mackinnon solutions.

2. `cvfw_ideal_tomography.py` (Code 2)
      Purpose: Simulates a heterogeneous 3D volume (e.g., slicing a sphere into multiple distinct layers).
      Physics: Also operates in the Ideal Regime. It dynamically loads a list of different morphological matrices and renders a full tomographic 3D hologram, proving that the algorithm can sustain asymmetric, arbitrary topologies without structural collapse.

  3. `cvfw_realistic_homogeneous.py` (Code 3)
      Purpose: Simulates a homogeneous 3D volume under Real Physical Conditions.
      Physics: Introduces the full coherent superposition of the volume. 
          Crosstalk: The electromagnetic interference between adjacent light sheets is active.
          Bessel-Root Anchoring: Employs an intelligent geometric shielding algorithm (`scipy.special.jn_zeros`) that mathematically locks the inter-planar spacing exactly at the dark rings of the J_0 Bessel function, mitigating out-of-plane destructive interference.
          Bessel-Gauss Apodization:   Applies a transverse Gaussian envelope (w_0) to restrict the infinite energy of ideal Bessel beams, mirroring realistic finite-aperture experimental laser setups.

  4. `cvfw_realistic_tomography.py` (Code 4)
      Purpose: The ultimate 3D volumetric display simulator.
      Physics: Combines the distinct tomographic slicing (from Code 2) with the rigorous real-world electrodynamics (from Code 3). It renders complex 3D structures (like hollow or solid spheres) subjected to severe, asymmetric cross-talk, utilizing root-anchoring and apodization to preserve the topological integrity of the optical trap.

Automation & Parameter Customization

The algorithms are designed to be highly modular. The entire physics engine is automated so that researchers can seamlessly alter the target geometries and wave properties. The core inputs are mapped as follows:

  1. The Morphological Function: F(x,z)
      How to customize: The user simply provides the path to a binary or grayscale image (e.g., `"F=MA.png"` or a list `["fatia_1.png", "fatia_2.png", ...]`).
      Automation: The algorithm automatically reads the image, resizes it to match the strict mathematical mesh mandated by the optical bandwidth (K_{band} = 2k), normalizes the intensities, and applies a smoothing Gaussian filter (sigma = 0.8) to prevent infinite-frequency singularities at sharp edges.

  2. Angular Frequency & Wavelength:
      How to customize: Modify the `lambda_0` variable (default is  632.8  nm).
      Automation: The script recalculates the total wavenumber k = 2\pi/\lambda_0, the spatial bandwidth, and the longitudinal carrier parameter  Q  (set to  0.80k  by default to ensure purely propagating, non-evanescent modes).

  3. Number of Parallel Filaments: P
      How to customize: Dictated by the transverse window R (default  30 \mu m ).
      Automation: The user  does not  need to manually guess how many threads are needed. The code calculates the minimum safe distance  \Delta x_{min} = 4.81 / k_{\rho, Q}  (the Rayleigh criterion for Bessel modes) and automatically discretizes the space into nx = P parallel, non-interfering threads.

  4. Longitudinal Spectral Sampling: N
      How to customize: Governed by the propagation length L (default  100 \mu m ).
      Automation: The spatial frequency nodes z_{samples} are dynamically allocated based on the Nyquist-Bessel criteria, generating the exact  N  discrete evaluation planes needed to map the continuous Mackinnon integral into a high-fidelity rendering.

  5. Number of Light Sheets: J
      How to customize: Alter `n_folhas` (for Codes 1 & 3) or simply pass a longer list of images `lista_img_paths` (for Codes 2 & 4).
      Automation: The script automatically calculates the spatial limits (`limite_y`) and distributes the  J  planes symmetrically around the optical axis.

  6. Transverse Spacing & Coordinates:  \rho_{0p}, \phi_{0p} 
      How to customize: The inter-planar spacing `espaco_y` defines the distance between sheets.
      Automation (The Magic):   
        In the Ideal codes, `espaco_y` is hardcoded (e.g.,  35 \mu m ). 
        In the Realistic codes, the algorithm actively searches for the closest analytical root of the  J_0  Bessel function relative to the target distance. It then overwrites `espaco_y` with this exact sub-micrometer coordinate (`zero_exato / k_rho_Q`), completely automating the crosstalk mitigation strategy.
        The polar translations  (\rho_{0p}, \phi_{0p})  are managed implicitly by the Cartesian meshgrid `(X_grid - xp)` and `(plano_y - yp)`, transforming localized Bessel modes seamlessly into the global volumetric space.

Requirements
Ensure you have the following Python libraries installed:
    `numpy` (Matrix operations and physics calculations)
    `matplotlib` (3D Volumetric rendering and Tomography plots)
    `scipy` (Bessel functions  J_0, J_1  and Root calculation)
    `Pillow` (`PIL`) (Image processing for  F(x,z)  mapping)

Running the Codes
1. Place your target morphological images (e.g., `.png` files) in the root directory.
2. Ensure the image paths in the bottom execution blocks match your files.
3. Run the script via terminal or your preferred IDE:
   ```bash
   python cvfw_realistic_tomography.py
