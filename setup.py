from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        check_call("python fetch_pretrained_model.py", shell=True)
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        check_call("python fetch_pretrained_model.py", shell=True)
        install.run(self)
setup(
    name='automatic_wood_pith_detector',
    version='1.0.0',
    description=' Automatic Wood Pith Detection method over RGB images',
    url='https://github.com/hmarichal93/apd',
    author='Henry Marichal',
    author_email='hmarichal93@gmail.com',
    license='MIT',
    packages=['automatic_wood_pith_detector'],
    install_requires=['numpy == 1.25.0',
        'opencv-python == 4.7.0.72',
        'pandas ==  2.0.2',
        'matplotlib == 3.7.1',
        'pathlib == 1.0.1',
        'scikit-learn == 1.2.2',
        'Pillow == 9.5.0',
        'scikit-image == 0.22.0',
        'ultralytics == 8.3.13',
                      ],

    classifiers=[
        'Development Status :: 1 - Accept',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)