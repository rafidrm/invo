from setuptools import setup


setup(
        name='invo',
        version='1.0',
        description='A Generalized Inverse Optimization package.',
        author='Rafid Mahmood',
        author_email='rafid.mahmood@mail.utoronto.ca',
        license='MIT',
        packages=['invo'],
        install_requires=['numpy', 'cvxpy'],
        zip_safe=False)
