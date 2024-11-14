[PYTHON]
class QueryGenerator:
    def __init__(self, dialect):
        self.dialect = dialect

    def get_where_conditions(self, smth, table_name, factory, options=None, prepend=True):
        if options is None:
            options = {}

        result = None
        where = {}
        self = self

        if Array.isArray(table_name):
            table_name = table_name[0]
            if Array.isArray(table_name):
                table_name = table_name[1]

        options['prefix'] = prepend and table_name

        if smth and smth._isSequelizeMethod:  # Checking a property is cheaper than a lot of instanceof calls
            result = self.handle_sequelize_method(smth, table_name, factory, options, prepend)
        elif Utils.is_plain_object(smth):
            return self.where_items_query(smth, {
                'model': factory,
                'prefix': prepend and table_name
            })
        elif type(smth) == int:
            primary_keys = factory and factory.primary_key or []

            if len(primary_keys) > 0:
                # Since we're just a number, assume only the first key
                primary_keys = primary_keys[0]
            else:
                primary_keys = 'id'

            where[primary_keys] = smth

            return self.where_items_query(where, {
                'model': factory,
                'prefix': prepend and table_name
            })
        elif type(smth) == str:
            return self.where_items_query(smth, {
                'model': factory,
                'prefix': prepend and table_name
            })
        elif Buffer.isBuffer(smth):
            result = self.escape(smth)
        elif Array.isArray(smth):
            if len(smth) == 0:
                return '1=1'
            if Utils.can_treat_array_as_and(smth):
                smth = {'$and': smth}

                result = self.get_where_conditions(smth, table_name, factory, options, prepend)
            else:
                result = Utils.format(smth, self.dialect)
        elif smth is None:
            return self.where_items_query(smth, {
                'model': factory,
                'prefix': prepend and table_name
            })

        return result if result else '1=1'

    def boolean_value(self, value):
        return value
[/PYTHON]
