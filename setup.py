from setuptools import setup


setup(
        name='invo',
        version='0.3',
        description='A Generalized Inverse Optimization package.',
        author='Rafid Mahmood',
        author_email='rafid.mahmood@mail.utoronto.ca',
        url='https://github.com/rafidrm/invo',
        download_url='https://github.com/rafidrm/invo/archive/0.3.tar.gz',
        license='MIT',
        packages=['invo', 
            'invo.LinearModels',
            'invo.utils'],
        keywords=['optimization','inverse'],
        install_requires=['numpy', 'cvxpy'],
        zip_safe=False)
