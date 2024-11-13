<div class="<?php echo $sizeClass; ?> ls-flex-column ls-panelboxes-panelbox text-primary" >
    <div class="panel panel-primary panel-clickable ls-panelboxes-panelbox-inner selector__<?=str_replace(' ', '_', strtolower($title))?>"
        id="panel-<?php echo $position; ?>"
        data-url="<?php echo htmlspecialchars($url, ENT_QUOTES, 'UTF-8'); ?>"
        <?php if ($external): ?>
            data-target="_blank"
        <?php endif; ?>
    >
        <div class="panel-heading">
            <div class="panel-title h4"><?php echo htmlspecialchars($title, ENT_QUOTES, 'UTF-8'); ?></div>
        </div>
        <div class="panel-body">
            <div class="panel-body-ico">
                <span class="<?php echo htmlspecialchars(CHtml::encode($ico), ENT_QUOTES, 'UTF-8'); ?>" style="font-size: 4em">
                </span>
            </div>
            <div class="panel-body-link">
                <?php echo htmlspecialchars($description, ENT_QUOTES, 'UTF-8'); ?>
                <?php if ($external): ?>
                    &nbsp;<i class="fa fa-external-link"></i>
                <?php endif; ?>
            </div>
        </div>
    </div>
</div>
