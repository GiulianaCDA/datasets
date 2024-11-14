package io.dataease.dto.datasource;

import io.dataease.plugins.datasource.entity.JdbcConfiguration;
import org.apache.commons.lang3.StringUtils;

public class MysqlConfiguration extends JdbcConfiguration {

    private String driver = "com.mysql.jdbc.Driver";
    private String extraParams = "characterEncoding=UTF-8&connectTimeout=5000&useSSL=false&allowPublicKeyRetrieval=true&zeroDateTimeBehavior=convertToNull";

    public String getJdbc() {
        StringBuilder jdbcUrlBuilder = new StringBuilder("jdbc:mysql://");
        if (StringUtils.isNotEmpty(getHost())) {
            jdbcUrlBuilder.append(getHost().trim());
        } else {
            throw new IllegalArgumentException("Host cannot be empty");
        }
        
        if (getPort() != null) {
            jdbcUrlBuilder.append(":").append(getPort().toString().trim());
        } else {
            throw new IllegalArgumentException("Port cannot be empty");
        }
        
        if (StringUtils.isNotEmpty(getDataBase())) {
            jdbcUrlBuilder.append("/").append(getDataBase().trim());
        } else {
            throw new IllegalArgumentException("Database name cannot be empty");
        }
        
        if (StringUtils.isNotEmpty(extraParams)) {
            jdbcUrlBuilder.append("?").append(getExtraParams().trim());
        }
        
        return jdbcUrlBuilder.toString();
    }
}
