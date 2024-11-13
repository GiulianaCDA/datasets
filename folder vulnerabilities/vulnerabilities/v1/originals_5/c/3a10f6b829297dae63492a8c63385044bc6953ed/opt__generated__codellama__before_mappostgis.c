 
[PYTHON]
import os
import sys

from mslib import msio as msl

def get_layer_info(ms_file, layer_name):
    # Initialize the MSSQL object
    ms = msl.MSSQL()

    # Open the MSSQL file
    if not ms.open(ms_file):
        print("Failed to open MSSQL file")
        sys.exit(1)

    # Get the layer info
    layer_info = ms.getLayerInfo(layer_name)

    # Close the MSSQL file
    ms.close()

    return layer_info

def main():
    # Path to the MSSQL file
    ms_file = "path/to/mssql_file.ms"

    # Layer name
    layer_name = "layer1"

    # Get the layer info
    layer_info = get_layer_info(ms_file, layer_name)

    # Print the layer info
    print("Layer Name:", layer_info.name)
    print("Number of Features:", layer_info.numFeatures)
    print("Feature Type:", layer_info.featureType)
    print("Spatial Reference System ID:", layer_info.srsId)

if __name__ == "__main__":
    main()
[/PYTHON]
[C++]
#include <mslib/msio.h>
#include <iostream>

int main()
{
    // Path to the MSSQL file
    const char* msFile = "path/to/mssql_file.ms";

    // Layer name
    const char* layerName = "layer1";

    // Initialize the MSSQL object
    MsSql ms;

    // Open the MSSQL file
    if (!ms.open(msFile)) {
        std::cout << "Failed to open MSSQL file" << std::endl;
        return 1;
    }

    // Get the layer info
    LayerInfo layerInfo = ms.getLayerInfo(layerName);

    // Print the layer info
    std::cout << "Layer Name: " << layerInfo.name << std::endl;
    std::cout << "Number of Features: " << layerInfo.numFeatures << std::endl;
    std::cout << "Feature Type: " << layerInfo.featureType << std::endl;
    std::cout << "Spatial Reference System ID: " << layerInfo.srsId << std::endl;

    // Close the MSSQL file
    ms.close();

    return 0;
}
[/C++]
