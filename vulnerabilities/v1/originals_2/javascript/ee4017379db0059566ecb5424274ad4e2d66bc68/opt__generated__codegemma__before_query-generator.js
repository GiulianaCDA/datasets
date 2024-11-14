const Utils = require('./utils');

const QueryGenerator = {
  generateQuery: function(model, options) {
    return this.getWhereConditions(model.where, model.name, model, options);
  },

  whereItemsQuery: function(items, options) {
    if (!items) return '1=1';

    return Object.keys(items).map(key => {
      return this.compileItem(key, items[key], options);
    }).join(' AND ');
  },

  compileItem: function(key, value, options) {
    var field = options.model.fields[key];

    if (!field) return value;

    var isVirtualField = field.virtual;

    if (isVirtualField && field.virtualOptions.type === 'function') {
      return value;
    }

    var comparator = field.comparator || '=';
    var escapeOptions = {
      escapeString: field.escape || true,
      escapeNumbers: field.escapeNumbers || false,
      escapeObjects: field.escapeObjects || false,
      escapeFunctions: field.escapeFunctions || false
    };

    if (typeof value === 'function') {
      return value(field);
    }

    if (field.type === 'array') {
      return this.getWhereConditions(value, key, options.model, options, false);
    }

    if (field.type === 'object' && field.model) {
      return this.getWhereConditions(value, field.model.name, field.model, options, false);
    }

    if (field.type === 'date' || field.type === 'timestamp') {
      if (value instanceof Date) {
        value = value.toISOString();
      } else if (typeof value === 'string') {
        value = "'" + value + "'";
      } else if (typeof value === 'number') {
        value = value.toString();
      } else {
        value = 'NULL';
      }
    }

    if (field.type === 'boolean') {
      value = this.booleanValue(value);
    }

    if (field.type === 'enum') {
      if (Array.isArray(value)) {
        value = '(' + value.map(v => "'" + v + "'").join(', ') + ')';
      } else {
        value = "'" + value + "'";
      }
    }

    if (value === undefined || value === null) {
      value = 'NULL';
    } else if (typeof value === 'string') {
      value = "'" + this.escape(value, field, escapeOptions) + "'";
    }

    if (comparator === 'IS NOT NULL' || comparator === 'IS NULL') {
      return [key, value].join(' ' + comparator + ' ');
    }

    if (comparator === 'LIKE' || comparator === 'NOT LIKE') {
      if (typeof value === 'string') {
        value = "'" + this.escape(value, field, escapeOptions) + "'";
      }
    }

    return [key, value].join(' ' + comparator + ' ');
  },

  escape: function(value, field, escapeOptions) {
    if (field.escape === false) return value;

    if (escapeOptions.escapeString && typeof value === 'string') {
      return value.replace(/"/g, '\\"');
    }

    if (escapeOptions.escapeNumbers && typeof value === 'number') {
      return value.toString();
    }

    if (escapeOptions.escapeObjects && typeof value === 'object') {
      return JSON.stringify(value);
    }

    if (escapeOptions.escapeFunctions && typeof value === 'function') {
      return value.toString();
    }

    return value;
  },

  handleSequelizeMethod: function(method) {
    return method.call(this);
  },

  getWhereConditions: function(smth, tableName, factory, options, prepend) {
    // ... existing getWhereConditions implementation
  },

  compileItem: function(key, value, options) {
    // ... existing compileItem implementation
  },

  escape: function(value, field, escapeOptions) {
    // ... existing escape implementation
  },

  handleSequelizeMethod: function(method) {
    // ... existing handleSequelizeMethod implementation
  },

  booleanValue: function(value) {
    // ... existing booleanValue implementation
  }
};

module.exports = QueryGenerator;
