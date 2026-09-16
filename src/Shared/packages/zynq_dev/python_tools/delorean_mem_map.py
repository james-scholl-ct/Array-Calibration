import jinja2
import os
import textwrap
import yaml

_MAP_YAML_FILE = os.path.join(os.path.dirname(__file__),
                              'yaml',
                              'delorean_axi_map.yml')

_JINJA2_TEMPLATE = textwrap.dedent('''
    {%- for periph, p in map.fields.items() | sort(attribute='0') -%}
    //  - {{ periph | upper }} registers/fields{{ '\n' -}}
    {%- for field, f in p.items() | sort(attribute='1.offset') -%}
    #define {{ periph | upper }}_{{ field | upper }}_WOFFSET {{
    ' ' * (44 - 18 - periph | length - field | length) }}{{
    "%d" | format(f.offset / 4) }}{{ '\n' -}}
    #define {{ periph | upper }}_{{ field | upper }}_POS     {{
    ' ' * (44 - 18 - periph | length - field | length) }}{{
    "%d" | format(f.pos) }}{{ '\n' -}}
    #define {{ periph | upper }}_{{ field | upper }}_MASK    {{
    ' ' * (44 - 18 - periph | length - field | length) }}0x{{
    "%08x" | format(map.get_field_mask(periph, field)) }}{{ '\n' -}}
    {% endfor -%}
    {{ '\n' if not loop.last -}}
    {% endfor -%}
''')

class DeloreanMemMap:
    def __init__(self, map_yaml=_MAP_YAML_FILE):
        self.map = self.parse_map_yaml(map_yaml)

    def parse_map_yaml(self, map_file):
        with open(map_file, 'r') as f:
            map_ = yaml.safe_load(f)
        return map_

    def __getitem__(self, item):
        '''Allows indexing on the instance to be an alias for indexing on the
        instance's map attribute.
        '''
        return self.map[item]

    def emit_c_headers(self):
        t = jinja2.Template(_JINJA2_TEMPLATE)
        out = t.render(map=self)
        return(out)

    def get_field_addr(self, periph, field):
        return (self.map['base_addrs'][periph] +
                self.map['fields'][periph][field]['offset'])

    def get_field_word_addr(self, periph, field):
        return self.map['fields'][periph][field]['offset'] >> 2

    def get_field_mask(self, periph, field, value=None):
        field_dict = self.map['fields'][periph][field]
        mask = (1 << field_dict['size']) - 1
        if value is None:
            return mask << field_dict['pos']
        else:
            return (value & mask) << field_dict['pos']

    def get_field_value(self, periph, field, data=0):
        field_dict = self.map['fields'][periph][field]
        mask = (1 << field_dict['size']) - 1
        return (data >> field_dict['pos']) & mask

    #def get_field_rd_mod_wr(self, periph, field):
    #    return self.map['fields'][periph][field].get('rd_mod_wr', False)

    #def get_set_field(self, periph, field):
    #    if self.get_field_has_rmw(periph, field):
    #        msg = "The field {}.{} is packed with other fields in the same "\
    #              "word and should not be written with this method."
    #        msg = msg.format(periph, field)
    #        raise RuntimeError(msg)

    #def lookup_field_mnemonic(self, periph, field, value):
    #    return self.map['fields'][periph][field]['mnemonic'][value]


if __name__ == '__main__':
    print(DeloreanMemMap().emit_c_headers())
