from setuptools import setup, find_packages

setup(
	name='GeneTEA',
	version='1.0.0',
	author="BroadInstitute CDS",
	description="NLP based method for gene set overrepresentation analysis",
	packages=find_packages(),
	package_data={'': ['*.r']}
)