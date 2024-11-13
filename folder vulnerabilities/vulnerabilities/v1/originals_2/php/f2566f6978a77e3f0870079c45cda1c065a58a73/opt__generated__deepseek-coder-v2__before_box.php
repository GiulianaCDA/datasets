<div class="<?php echo htmlspecialchars($sizeClass, ENT_QUOTES); ?> ls-flex-column ls-panelboxes-panelbox text-primary">
    <div class="panel panel-primary panel-clickable ls-panelboxes-panelbox-inner selector__<?=str_replace(' ', '_', strtolower(htmlspecialchars($title, ENT_QUOTES))); ?>"
        id="panel-<?php echo htmlspecialchars($position, ENT_QUOTES); ?>"
        data-url="<?php echo htmlspecialchars($url, ENT_QUOTES); ?>"
        <?php if ($external): ?>
            data-target="_blank"
        <?php endif; ?>
    >
        <div class="panel-heading">
            <div class="panel-title h4"><?php eT(htmlspecialchars($title, ENT_QUOTES)); ?></div>
        </div>
        <div class="panel-body">
            <div class="panel-body-ico">
                <span class="sr-only"><?php eT(htmlspecialchars($title, ENT_QUOTES)); ?></span>
                <span class="<?php echo htmlspecialchars($ico, ENT_QUOTES); ?>" style="font-size: 4em"></span>
            </div>
            <div class="panel-body-link">
                <?php eT(htmlspecialchars($description, ENT_QUOTES)); ?>
                <?php if ($external): ?>
                    &nbsp;<i class="fa fa-external-link"></i>
                <?php endif; ?>
            </div>
        </div>
    </div>
</div>
