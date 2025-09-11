from setuptools import setup, find_packages
import os


# Read the requirements from the requirements.txt file.
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        # Read each line and filter out empty lines and comments.
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#') and not ("--" in line)
        ]
    return requirements


setup(
    name='Models',
    version='0.1.0',
    author='Bradley Odimmasi',
    author_email='bodimmasi@students.uonbi.ac.ke',
    description='Random Modules that can be used in various projects',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/KshRaven/',
    packages=find_packages(),
    install_requires=load_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
