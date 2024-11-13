// Create a new ELFOBJ from a file
ELFOBJ* elf = Elf_(r_bin_elf_new) ("binary.elf", false);

// Create a new ELFOBJ from an RBuffer object
RBuffer buffer;
// ... initialize buffer ...
ELFOBJ* elf_buf = Elf_(r_bin_elf_new_buf) (&buffer, false);

// Convert a physical address to a virtual address
ut64 virtual_address = Elf_(r_bin_elf_p2v) (elf, physical_address);

// Convert a virtual address to a relative physical address
ut64 relative_physical_address = Elf_(r_bin_elf_v2p) (elf, virtual_address);
