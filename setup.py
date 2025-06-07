from setuptools import setup
setup(
    name = 'python-db',
    packages = ['pydb'],
    version = "0.0.5",
    description = 'Placeholder description',
    author = "Jason Trigg",
    author_email = "jasontrigg0@gmail.com",
    url = "https://github.com/jasontrigg0/python-db",
    download_url = 'https://github.com/jasontrigg0/python-db/tarball/0.0.5',
    scripts=['bin/db'],
    install_requires=['argparse',
                      'jtutils',
                      'python-csv',
                      'python-service',
                      'mysqlclient'
    ],
    keywords = [],
    classifiers = [],
)
