RBinJavaObj *bin = r_bin_java_load_bin (buf, buf_sz);
ut32 idx = U(r_bin_java_get_cp_idx_with_name) (bin, "Hello", 5);
if (idx != 0) {
	const RBinJavaCPTypeObj *obj = r_bin_java_get_cp_item (bin, idx);
	printf ("Constant pool item %d: '%s'\n", idx, obj->bytes);
}
