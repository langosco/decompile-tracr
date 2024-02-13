import setuptools

setuptools.setup(name='decompile-tracr',
      version='0.0.1',
      description="Program Generator for RASP",
      packages=['decompile_tracr'],
      install_requires=[
            "chex",
            "flax",
            "jax",
            "jaxtyping",
            "networkx",
            "numpy",
            "setuptools",
            "tqdm",
            "dill",
      ],
)


