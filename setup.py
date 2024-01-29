import setuptools

setuptools.setup(name='rasp-generator',
      version='0.0.1',
      description="Program Generator for RASP",
      packages=['rasp_generator', 'rasp_tokenizer'],
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


