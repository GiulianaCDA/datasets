int main() {
    // Attach to a running child process with PID 1234
    int pid = 1234;

    // Send a SIGUSR1 signal to the child process
    ptrace(PTRACE_CONT, pid, 0, SIGUSR1);

    // Read the value of a memory address in the child process
    int value;
    ptrace(PTRACE_PEEKTEXT, pid, memory_address, &value);

    // Close the child process
    ptrace(PTRACE_DETACH, pid, 0, 0);

    return 0;
}
