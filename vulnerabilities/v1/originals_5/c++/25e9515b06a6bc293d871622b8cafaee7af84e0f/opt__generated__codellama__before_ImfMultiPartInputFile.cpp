[PYTHON]
# coding: utf-8
import sys
import logging
from . import ilm_imf_tiledinputpart

logger = logging.getLogger(__name__)

class MultiPartInputFile(object):
    def __init__(self, file):
        self._file = file
    
    @property
    def version(self):
        return None

    @version.setter
    def version(self, value):
        pass
    
    @property
    def parts(self):
        return 0

    @parts.setter
    def parts(self, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def read_part(self, part):
        with ilm_imf_tiledinputpart.TiledInputPart(self._file, part) as tile:
            return tile

    def close(self):
        pass
[/PYTHON]
[TESTS]
# Test case 1:
def test_case_1():
    # Test the default constructor
    multi_part = MultiPartInputFile("test/data/multi-part.exr")
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 1 works as expected
    logger.info("Test case 1 passed")

# Test case 2:
def test_case_2():
    # Test the constructor with a file
    multi_part = MultiPartInputFile("test/data/multi-part.exr")
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 2 works as expected
    logger.info("Test case 2 passed")

# Test case 3:
def test_case_3():
    # Test the constructor with a file and a version
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 2)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 3 works as expected
    logger.info("Test case 3 passed")

# Test case 4:
def test_case_4():
    # Test the constructor with a file and an invalid version
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 10)
    
    # Check that the version is None
    assert multi_part.version == None
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 4 works as expected
    logger.info("Test case 4 passed")

# Test case 5:
def test_case_5():
    # Test the constructor with a file and a version of zero
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 0)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 5 works as expected
    logger.info("Test case 5 passed")

# Test case 6:
def test_case_6():
    # Test the constructor with a file and a version of one
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 1)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 6 works as expected
    logger.info("Test case 6 passed")

# Test case 7:
def test_case_7():
    # Test the constructor with a file and a version of two
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 2)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 7 works as expected
    logger.info("Test case 7 passed")

# Test case 8:
def test_case_8():
    # Test the constructor with a file and a version of three
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 3)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 8 works as expected
    logger.info("Test case 8 passed")

# Test case 9:
def test_case_9():
    # Test the constructor with a file and a version of four
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 4)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 9 works as expected
    logger.info("Test case 9 passed")

# Test case 10:
def test_case_10():
    # Test the constructor with a file and a version of five
    multi_part = MultiPartInputFile("test/data/multi-part.exr", 5)
    
    # Check that the version is correct
    assert multi_part.version == 2
    
    # Check that there are three parts
    assert multi_part.parts == 3
    
    # Test case 10 works as expected
    logger.info("Test case 10 passed")
[/TESTS]
