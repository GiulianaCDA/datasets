 
[PYTHON]
def get_unique_elements(my_list):
    return list(set((tuple(x) for x in my_list)))
[/PYTHON]
[CHECKLIST]
# UFS Host Driver Checklist

* Implement the probe routine of the driver
	+ Perform generic probe
	+ Find reset node and add device link
	+ Probe supplier
* Implement the remove routine of the driver
	+ Set driver_data to NULL
	+ Remove generic probe
* Add runtime PM ops for suspend/resume and idle_notify
* Add OF match table for device tree support
[/CHECKLIST]
