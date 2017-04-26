from setuptools import setup


setup(
        name='invo',
        version='0.1',
        description='A Generalized Inverse Optimization package.',
        author='Rafid Mahmood',
        author_email='rafid.mahmood@mail.utoronto.ca',
        url='https://github.com/rafidrm/invo',
        download_url='https://github.com/rafidrm/invo/archive/0.1.tar.gz',
        license='MIT',
        packages=['invo'],
        keywords=['optimization','inverse'],
        install_requires=['numpy', 'cvxpy'],
        zip_safe=False)
