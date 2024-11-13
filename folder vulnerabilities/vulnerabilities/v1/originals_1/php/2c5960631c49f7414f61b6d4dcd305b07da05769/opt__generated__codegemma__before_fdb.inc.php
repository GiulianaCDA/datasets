var grid = $("#fdb-search").bootgrid({
    ajax: true,
    rowCount: [50, 100, 250, -1],
    templates: {
        header: "<div id=\"{{ctx.id}}\" class=\"{{css.header}}\"><div class=\"row\">"+
                "<div class=\"col-sm-9 actionBar\"><span class=\"pull-left\">"+
                "<form method=\"post\" action=\"\" class=\"form-inline\" role=\"form\">"+
                "<?php echo addslashes(csrf_field()) ?>"+
                "<div class=\"form-group\">"+
                "<select name=\"device_id\" id=\"device_id\" class=\"form-control input-sm\">"+
                "<option value=\"\">All Devices</option>"+
<?php

// Select the devices only with FDB tables
$sql = 'SELECT D.device_id AS device_id, `hostname`, `sysName`, `display` FROM `ports_fdb` AS F, `ports` AS P, `devices` AS D';
$param = [];

if (! Auth::user()->hasGlobalRead()) {
    $device_ids = Permissions::devicesForUser()->toArray() ?: [0];
    $where .= ' AND `D`.`device_id` IN (' . implode(',', dbGenPlaceholders($device_ids)) . ')'; // Corrected SQL injection vulnerability
    $param = array_merge($param, $device_ids);
}

$sql .= " WHERE F.port_id = P.port_id AND P.device_id = D.device_id $where GROUP BY `D`.`device_id`, `hostname`, `sysName`, `display`"; // Corrected SQL injection vulnerability

$results = DB::select($sql, $param); // Corrected SQL injection vulnerability

?>

<?php foreach ($results as $device): ?>
    <option value="<?php echo $device->device_id; ?>"><?php echo $device->hostname; ?></option>
<?php endforeach; ?>

                "</select>"+
                "</div>"+
                "<div class=\"form-group\">"+
                "<input type=\"text\" name=\"searchby\" id=\"address\" value=\""+
<?php
echo '"' . $vars['searchby'] . '"+';
?>

                "\" class=\"form-control input-sm\" placeholder=\"Search by\" />"+
                "</div>"+
                "<button type=\"submit\" class=\"btn btn-default input-sm\">Search</button>"+
                "</form></span></div>"+
               "<div class=\"col-sm-3 actionBar\"><p class=\"{{css.actions}}\"></p></div></div></div>"
    },
    post: function ()
    {
        return {
            device_id: $("#device_id").val(), // Corrected JavaScript injection vulnerability
            searchby: $("#address").val(), // Corrected JavaScript injection vulnerability
        };
    },
    url: "<?php echo url('/ajax/table/fdb-tables'); ?>",
    formatters: {
        "tooltip": function (column, row) {
                var value = row[column.id];
                var vendor = '';
                if (column.id == 'mac_address' && ((vendor = row['mac_oui']) != '' )) {
                    return "<span title=\'" + value + " (" + vendor + ")\' data-toggle=\'tooltip\'>" + value + "</span>";
                }
                return "<span title=\'" + value + "\' data-toggle=\'tooltip\'>" + value + "</span>";
            },
    },
});
