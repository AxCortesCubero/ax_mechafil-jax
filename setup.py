import setuptools

setuptools.setup(
    name="mechafil_jax",
    version="0.1",
    packages=["mechafil_jax"],
    install_requires=[
        "jax", 
        "jaxlib", 
        "numpy", 
        "scipy", 
        "matplotlib",
        "pystarboard @ git+https://git@github.com/celtd/pystarboard.git",
    ],

    tests_require = [
        'pytest',
        'mechaFIL @ git+https://git@github.com/celtd/filecoin-mecha-twin.git@mechafil_jax'  # get the branch that is built for comparisons w/ jax
    ]
)
