int madvise(unsigned long start, size_t len, int behavior) {
  return syscall(SYS_madvise, start, len, behavior);
}
