 /**
     * Returns content object data_map array. Array is build using following pattern:
     * - data_map[attribute_identifier]['type']           : attribute type string
     * - data_map[attribute_identifier]['identifier']     : attribute identifier string
     * - data_map[attribute_identifier]['content']        : mixed ( content )
     * - image_attributes                                 : array of all images attributes in object
     *   if is images attribute and content is set then value is an array with following keys:
     *      - original
     *      - aliasName1
     *      - aliasName2
     *      - ...
     *      - aliasNameN
     *  if $params['generateImageAliases'] = true then all aliases are pre generated and stored in data_map.
     *  if $params['generateImageAliases'] = false then only original image is stored in data_map and
     *   other aliases can be generated later with eZContentObject::attribute( $attributeIdentifier )->value()->aliases() method
     *
     * @param eZContentObject $object
     * @param array           $params (optional) - array of parameters that can influence return value. Available params are:
     *   - imageDataTypes          : array with all data type names that will be treated as images (default: array('ezimage'))
     *   - datatypeBlacklist        : array with all datatype names to ignore (no content is returned for these) (default: array('eztags', 'ezuser'))
     *   - imageSizes              : array of alias names that should be used if generateImageAliases = true (default: array('small', 'medium', 'large', 'original' ))
     *   - imagePreGenerateSizes    : array with aliases from $imageSizes that will be generated on the fly and stored in data_map. All other alias names will not be pre-generated.
     *                               If $generateImageAliases = false then this parameter is ignored (default: array('medium', 'large'))
     *   - generateImageAliases     : if true all images aliases from $imageSizes will be generated and stored in data_map on the fly. If false only original image alias will be pre-generated, other aliases can be generated later with eZContentObject::attribute( $attributeIdentifier )->value()->aliases() method (default: true)
     *   - ignoreVisibility         : if true then object attributes that are not visible to current user won't be ignored. If false then those attributes will be returned as empty values ( default: false )
     * @return array
    */
    public static function dataMap( eZContentObject $object, $params = array() )
    {
        $ret                            = array();
        $attrtibuteArray                = array();
        if ( !is_array( $params ) || count( $params ) == 0)
        {
            $params['imageDataTypes']          = array('ezimage');
            $params['datatypeBlacklist']        = array('eztags', 'ezuser');
            $params['imageSizes']               = array('small', 'medium', 'large', 'original' );
            $params['imagePreGenerateSizes']    = array('medium', 'large');
            $params['generateImageAliases']     = true;
            $params['ignoreVisibility']         = false;
        }
        else
        {
            if ( !isset( $params['imageDataTypes'] ) || !is_array( $params['imageDataTypes'] ) )
                $params['imageDataTypes']          = array('ezimage');

            if ( !isset( $params['datatypeBlacklist'] ) || !is_array( $params['datatypeBlacklist'] ) )
                $params['datatypeBlacklist']        = array('eztags', 'ezuser');

            if ( !isset( $params['imageSizes'] ) || !is_array( $params['imageSizes'] ) )
                $params['imageSizes']               = array('small', 'medium', 'large', 'original' );

            if ( !isset( $params['imagePreGenerateSizes'] ) || !is_array( $params['imagePreGenerateSizes'] ) )
                $params['imagePreGenerateSizes']    = array('medium', 'large');

            if ( !isset( $params['generateImageAliases'] ) || ( $params['generateImageAliases'] != true && $params['generateImageAliases'] != false) )
                $params['generateImageAliases']     = true;

            if ( !isset( $params['ignoreVisibility'] ) || ( $params['ignoreVisibility'] != true && $params['ignoreVisibility'] != false) )
                $params['ignoreVisibility']         = false;
        }

        $dataMap = $object->attribute('data_map');

        if ( $params['generateImageAliases'] == true )
            $imageSizeParams                   = array( 'alias' => implode( ',', $params['imageSizes']) );
        else
            $imageSizeParams                   = null;

        foreach ($dataMap as $attributeIdentifier=>$attributes) {
            if ( !in_array($attributeIdentifier, $params['datatypeBlacklist'] ) )
            {
                foreach ($attributes as $atrributeIndex=>$attributeData) {
                    //ignore visibility or not?
                    if ( $params['ignoreVisibility'] == true || $attributeData['visible'] == true )
                        $atr = $object->attribute($attributeIdentifier, $atrributeIndex);
                    else
                        $atr = null;
                }
            }

            if ( in_array($attributes[0]['data_type_string'], $params['imageDataTypes'] ) )
                $ret['image_attributes'][]   = $attributeIdentifier;

            //set value and type for attribute
            if ($atr !== null) {
            	if( is_object( $atr->value() ) && get_class($atr->value()) == 'eZImageAttributeContent' )
                    $attrtibuteArray[$attributeIdentifier]['content'] = $atr->value()->imageAliases( $imageSizeParams );
                else
                    $attrtibuteArray[$attributeIdentifier]['content'] = $atr->value();
            }
            else {
            	$attrtibuteArray[$attributeIdentifier]['content'] = null;
            }
            //set attribute type and identifier
            $attrtibuteArray[$attributeIdentifier]['type']           = $attributes[0]['data_type_string'];
            $attrtibuteArray[$attributeIdentifier]['identifier']     = $attributeIdentifier;
        }

        $ret['data_map'] = $attrtibuteArray;

        return $ret;
    }