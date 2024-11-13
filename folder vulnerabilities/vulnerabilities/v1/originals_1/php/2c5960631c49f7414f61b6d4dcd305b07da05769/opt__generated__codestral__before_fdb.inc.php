<script>
var grid = $("#fdb-search").bootgrid({
    ...
    post: function () {
        return {
            device_id: encodeURIComponent('<?php echo $vars['device_id']; ?>'),
            searchby: encodeURIComponent('<?php echo $vars['searchby']; ?>'),
            searchPhrase: encodeURIComponent('<?php echo $vars['searchPhrase']; ?>'),
            dns: $("#fdb-search").bootgrid("getColumnSettings").find(col => col.id === "dnsname").visible,
        };
    },
    ...
});
</script>
