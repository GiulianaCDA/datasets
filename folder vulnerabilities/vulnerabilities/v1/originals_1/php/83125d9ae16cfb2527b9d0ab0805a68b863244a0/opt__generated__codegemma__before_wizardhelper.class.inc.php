public function GetIdForField($sFieldName)
{
    $sResult = '';
    if (isset($this->m_aData['m_oFieldsMap'][$sFieldName]))
    {
        $sResult = $this->m_aData['m_oFieldsMap'][$sFieldName];
    }
    return $sResult;
}
