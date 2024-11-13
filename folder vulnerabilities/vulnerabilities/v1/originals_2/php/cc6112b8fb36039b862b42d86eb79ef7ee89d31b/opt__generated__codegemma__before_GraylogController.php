<?php

namespace App\Http\Controllers\Table;

use App\ApiClients\GraylogApi;
use App\Models\Device;
use DateInterval;
use DateTime;
use DateTimeZone;
use Illuminate\Http\Request;
use LibreNMS\Config;
use LibreNMS\Util\Url;

class GraylogController extends SimpleTableController
{
    private $timezone;
    private $deviceCache = [];

    public function __construct()
    {
        $timezone = Config::get('graylog.timezone');
        $this->timezone = $timezone ? new DateTimeZone($timezone) : null;
    }

    public function __invoke(Request $request, GraylogApi $api)
    {
        if (! $api->isConfigured()) {
            return response()->json([
                'error' => 'Graylog is not configured',
            ], 503);
        }

        $this->validate($request, [
            'stream' => 'nullable|alpha_num',
            'device' => 'nullable|integer', // Corrected data type
            'range' => 'nullable|integer', // Corrected data type
            'loglevel' => 'nullable|integer|min:0|max:7',
        ]);

        $search = $request->get('searchPhrase');
        $device_id = $request->get('device');
        $device = $device_id ? Device::find($device_id) : null;
        $range = $request->get('range', 0);
        $limit = $request->get('rowCount', 10);
        $page = $request->get('current', 1);
        $offset = ($page - 1) * $limit;
        $loglevel = $request->get('loglevel') ?? Config::get('graylog.loglevel');

        $query = $api->buildSimpleQuery($search, $device) .
            ($loglevel !== null ? ' AND level: <=' . $loglevel : '');

        $sort = null;
        foreach ($request->get('sort', []) as $field => $direction) { // Corrected field and direction access
            $sort .= $field . ':' . $direction . ',';
        }
        $sort = rtrim($sort, ',');

        return $this->getData($api, $query, $sort, $limit, $offset);
    }

    private function getData($api, $query, $sort, $limit, $offset)
    {
        try {
            $response = $api->getData($query, $sort, $limit, $offset);
            return response()->json($response);
        } catch (\Exception $e) {
            return response()->json(['error' => 'Failed to fetch data from Graylog'], 500);
        }
    }

    private function severityLabel($severity)
    {
        $map = [
            '0' => 'label-danger',
            '1' => 'label-danger',
            '2' => 'label-danger',
            '3' => 'label-danger',
            '4' => 'label-warning',
            '5' => 'label-info',
            '6' => 'label-info',
            '7' => 'label-default',
            ''  => 'label-info',
        ];
        $barColor = isset($map[$severity]) ? $map[$severity] : 'label-info';

        return '<span class="alert-status ' . $barColor . '" style="margin-right:8px;float:left;"></span>';
    }

    /**
     * Cache device lookups so we don't lookup for every entry
     *
     * @param  mixed  $source
     * @return mixed
     */
    private function deviceFromSource($source)
    {
        if (! isset($this->deviceCache[$source])) {
            $this->deviceCache[$source] = Device::findByIp($source) ?: Device::findByHostname($source);
        }

        return $this->deviceCache[$source];
    }
}
