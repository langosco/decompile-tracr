import setuptools

setuptools.setup(name='rasp-gen',
      version='0.0.1',
      description="Program Generator for RASP",
      packages=['rasp_gen'],
      install_requires=[
            "chex",
            "flax",
            "jax",
            "jaxtyping",
            "networkx",
            "numpy",
            "setuptools",
            "tqdm",
            "h5py",
            "matplotlib",
            "psutil",
      ],
)


