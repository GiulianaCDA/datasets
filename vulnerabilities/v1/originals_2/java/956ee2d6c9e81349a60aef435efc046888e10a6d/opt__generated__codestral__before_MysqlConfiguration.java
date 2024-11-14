import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public String getJdbc() {
    StringBuilder jdbcUrl = new StringBuilder("jdbc:mysql://");
    jdbcUrl.append(URLEncoder.encode(getHost().trim(), StandardCharsets.UTF_8));
    jdbcUrl.append(":").append(URLEncoder.encode(getPort().toString().trim(), StandardCharsets.UTF_8));
    jdbcUrl.append("/").append(URLEncoder.encode(getDataBase().trim(), StandardCharsets.UTF_8));

    if(!StringUtils.isEmpty(extraParams.trim())){
        jdbcUrl.append("?").append(URLEncoder.encode(getExtraParams().trim(), StandardCharsets.UTF_8));
    }

    return jdbcUrl.toString();
}
