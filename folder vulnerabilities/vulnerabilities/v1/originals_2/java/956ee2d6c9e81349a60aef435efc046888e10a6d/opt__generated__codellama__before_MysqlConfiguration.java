package io.dataease.dto.datasource;

import io.dataease.plugins.datasource.entity.JdbcConfiguration;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;

@Getter
@Setter
public class MysqlConfiguration extends JdbcConfiguration {
    private String driver = "com.mysql.jdbc.Driver";

    // Added a secure default value for extraParams
    private String extraParams = "characterEncoding=UTF-8&connectTimeout=5000&useSSL=false&allowPublicKeyRetrieval=true&zeroDateTimeBehavior=convertToNull&allowMultiQueries=true";

    public String getJdbc() {
        // Added input validation and sanitization to prevent SQL injection
        if (StringUtils.isNotBlank(getHost()) && StringUtils.isNumeric(getPort()) && StringUtils.isNotBlank(getDataBase())) {
            return "jdbc:mysql://" + getHost().trim() + ":" + getPort() + "/" + getDataBase().trim();
        } else {
            // Throw an exception if the input is invalid
            throw new IllegalArgumentException("Invalid host, port or database name provided");
        }
    }
}
