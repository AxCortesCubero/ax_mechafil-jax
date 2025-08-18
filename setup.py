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
        "scenario-generator @ git+https://github.com/CELtd/scenario-generator", 
        "pystarboard @ git+https://github.com/CELtd/pystarboard@updated_dependencies",

    ],
    extras_require = {
    "test": [
        "pytest",
        "mechaFIL @ git+https://github.com/CELtd/filecoin-mecha-twin@mechafil-jax-pandas-updates",
    ]
}
)