from setuptools import find_packages, setup
from typing import List


def  get_requirements(file_path:str) ->List[str]:
    """find packages from requirements.txt"""

    with open(file_path) as f:
        requirements = f.readline()
        # remove line break, comma, and e
        requirements =  [r.strip() for r in  requirements if r.strip() and r.strip != '-e']
        return requirements

set(
    name = "election prediction model",
    version = '0.0.1',
    author = 'xyz',
    author_email = 'eutjfne554@gmail.com',
    packages = find_packages(),
    install_require = get_requirements('requirements.txt')
)