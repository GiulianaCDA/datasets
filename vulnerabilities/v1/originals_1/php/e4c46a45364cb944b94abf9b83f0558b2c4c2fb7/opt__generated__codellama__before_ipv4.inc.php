<div class="panel panel-default panel-condensed">
    <div class="panel-heading">
        <strong>IPv4 Addresses</strong>
    </div>
    <table id="ipv4-search" class="table table-hover table-condensed table-striped">
        <thead>
            <tr>
                <th data-column-id="hostname" data-order="asc">Device</th>
                <th data-column-id="interface">Interface</th>
                <th data-column-id="address" data-sortable="false" data-formatter="tooltip">Address</th>
                <th data-column-id="description" data-sortable="false" data-formatter="tooltip">Description</th>
            </tr>
        </thead>
    </table>
</div>

<script type="text/javascript">
$("#ipv4-search").bootgrid({
    formatters: {
        "tooltip": function (column, row) {
            var value = row[column.id];
            return "<span title=\'" + value + "\' data-toggle=\'tooltip\'>" + value + "</span>";
        },
    },
}).on("loaded.rs.jquery.bootgrid", function () {
    $("#ipv4-search").find(".command-delete").prop("disabled", true);
});
</script>
