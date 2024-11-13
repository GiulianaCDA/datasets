// Export a content object item with ID 123
$params = [
    // ... parameters
];
$itemExportData = eZContentObjectItemExport::exportContentObjectItem( $params );

// Generate XML from the data map
$xmlString = eZContentObjectItemExport::xmlEncode( $itemExportData['data_map'] );
