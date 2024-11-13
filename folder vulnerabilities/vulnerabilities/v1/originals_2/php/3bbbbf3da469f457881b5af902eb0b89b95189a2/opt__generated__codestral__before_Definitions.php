<?php
namespace October\Rain\Filesystem;

use Config;
use Exception;

class Definitions
{
    // ... other methods ...

    /**
     * Determines if a path should be ignored, sourced from the ignoreFiles and ignorePatterns definitions.
     * @param string $path Specifies a path to check.
     * @return boolean Returns TRUE if the path is visible.
     */
    public static function isPathIgnored($path)
    {
        $ignoreNames = self::get('ignoreFiles');
        $ignorePatterns = self::get('ignorePatterns');

        if (in_array($path, $ignoreNames)) {
            return true;
        }

        // Update to avoid using regular expressions for simple string matching.
        foreach ($ignorePatterns as $pattern) {
            if (strpos($path, $pattern) === 0) {
                return true;
            }
        }

        return false;
    }

    // ... other methods ...

    /**
     * File patterns that can be safely ignored.
     * This list can be customized with config:
     * - cms.fileDefinitions.ignorePatterns
     */
    protected function ignorePatterns()
    {
        return [
            '.' // Changed to match any path starting with a dot.
        ];
    }
}
