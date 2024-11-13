    public function getItemConfig(\Model_CartProduct $model)
    {
        return $this->di['tools']->decodeJ($model->config);
    }
    