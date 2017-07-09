#! usr/bin/env/python
#
# By Rafid Mahmood <rafid.mahmood@mail.utoronto.ca>
# License: MIT


from setuptools import setup


setup(
        name='invo',
        version='0.4',
        description='A Generalized Inverse Optimization package.',
        author='Rafid Mahmood',
        author_email='rafid.mahmood@mail.utoronto.ca',
        url='https://github.com/rafidrm/invo',
        download_url='https://github.com/rafidrm/invo/archive/0.4.tar.gz',
        license='MIT',
        packages=['invo', 
            'invo.LinearModels',
            'invo.utils'],
        keywords=['optimization','inverse'],
        install_requires=['numpy', 'cvxpy'],
        zip_safe=False)
