from setuptools import find_packages, setup
from typing import List

HYPHEN_E = '-e .'
def get_requirement(file_path:str)->List[str]:
    #this function will return a list of requirements
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]  #List Comprehension

        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)
        
    return requirements

#This details acts as a metadata for the whole project
setup(
name= 'ML Project',
version='0.0.1',
author='DavidLuke',
packages=find_packages(),
install_requires=get_requirement('requirements.txt')
)
