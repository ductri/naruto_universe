from setuptools import setup, find_packages

setup(
    name='naruto_skills',
    version='1.2',
    description='Naruto has to do some tasks very quickly. The lib helps him to deal with those tremendous tasks',
    url='',
    author='Duc Tri Nguyen',
    author_email='ductricse@gmail.com',
    license='MIT',
    package_data={
        # And include any *.msg files found in the 'hello' package, too:
        '': ['*.txt']
    },
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    ],
    install_requires=['gensim>=3.8.1', 'numpy>=1.13.3', 'nltk>=3.2.4', 'sklearn']
)
