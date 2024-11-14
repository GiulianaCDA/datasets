<?php
$sizeClass = htmlspecialchars($sizeClass);
$title = htmlspecialchars($title, ENT_QUOTES, 'UTF-8');
$position = htmlspecialchars($position);
$url = htmlspecialchars($url);
$ico = htmlspecialchars($ico);
$description = htmlspecialchars($description, ENT_QUOTES, 'UTF-8');
?>
<div class="<?= $sizeClass ?> ls-flex-column ls-panelboxes-panelbox text-primary">
    <div class="panel panel-primary panel-clickable ls-panelboxes-panelbox-inner selector__<?= str_replace(' ', '_', strtolower($title))?>" id="panel-<?= $position ?>" data-url="<?= $url ?>" <?php if ($external): ?>data-target="_blank"<?php endif; ?>>
        <div class="panel-heading">
            <div class="panel-title h4"><?= html_entity_decode($title); ?></div>
        </div>
        <div class="panel-body">
            <div class="panel-body-ico">
                <span class="sr-only"><?= html_entity_decode($title); ?></span>
                <span class="<?= $ico ?>" style="font-size: 4em"></span>
            </div>
            <div class="panel-body-link">
                <?= html_entity_decode($description); ?>
                <?php if ($external): ?>&nbsp;<i class="fa fa-external-link"></i><?php endif; ?>
            </div>
        </div>
    </div>
</div>
