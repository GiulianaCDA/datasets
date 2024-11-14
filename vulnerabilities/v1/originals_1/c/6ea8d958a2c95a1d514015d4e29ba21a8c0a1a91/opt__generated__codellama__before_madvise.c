  This is a system call that allows the kernel to advise the kernel about how it should manage memory pages. The first three parameters of this function are the start address, length, and behavior of the madvice. The behavior can be one of several options:
  1. MADV_NORMAL : This is the default behavior where the system will use a page's physical location to determine whether it should be swapped out or not.
  2. MADV_RANDOM : This advice tells the kernel that the data is likely to be accessed randomly in the near future. The kernel can use this information to prioritize the page cache and optimize memory usage.
  3. MADV_SEQUENTIAL : This advice tells the kernel that the data will be accessed sequentially, which allows it to prefetch pages in advance and improve performance.
  4. MADV_WILLNEED : This advice tells the kernel that the page is likely to be needed soon, so it should not swap out the page cache for this page.
  5. MADV_DONTNEED : This advice tells the kernel that the page is no longer needed and can be swapped out if necessary.
  The return value of this function will either be zero (success) or an error code.
  
[PYTHON]
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/madvice/")
def read_item(behavior: str = Query(None)):
    return {"Behaviour": behavior}
 [/PYTHON]
[TESTS]
# Test case 1:
def test_case_1():
    assert madvice("start", "len_in", "MADV_NORMAL") == 0
# Test case 2:
def test_case_2():
    assert madvice("start", "len_in", "MADV_RANDOM") == 0
# Test case 3:
def test_case_3():
    assert madvice("start", "len_in", "MADV_SEQUENTIAL") == 0
# Test case 4:
def test_case_4():
    assert madvice("start", "len_in", "MADV_WILLNEED") == 0
# Test case 5:
def test_case_5():
    assert madvice("start", "len_in", "MADV_DONTNEED") == 0
[/TESTS]
