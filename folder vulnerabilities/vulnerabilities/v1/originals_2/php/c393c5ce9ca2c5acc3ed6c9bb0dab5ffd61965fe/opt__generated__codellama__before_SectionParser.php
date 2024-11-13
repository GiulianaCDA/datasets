[PYTHON]
import os
import sys

class IniParser:
    def parse(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            sections = content.split('==')
            return self.parse_sections(sections)

    def parse_sections(self, sections):
        count = len(sections)
        if count == 3:
            result = {'settings': sections[0], 'markup': sections[2]}
        elif count == 2:
            result = {'settings': sections[0], 'markup': sections[1]}
        elif count == 1:
            result = {'markup': sections[0]}
        return result

    def parse_offset(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            sections = content.split('==')
            count = len(sections)
            result = {'settings': None, 'code': None, 'markup': None}
            if count >= 3:
                result['settings'] = self.adjust_line_position(content)
                result['code'] = self.calculate_line_position(content)
                result['markup'] = self.calculate_line_position(content, 2)
            elif count == 2:
                result['settings'] = self.adjust_line_position(content)
                result['markup'] = self.calculate_line_position(content)
            elif count == 1:
                result['markup'] = 1
            return result

    def adjust_line_position(self, content, start_line=1):
        for line in content.splitlines():
            if line.strip() == '==':
                start_line += 1
        return start_line

    def calculate_line_position(self, content, instance=1):
        count = 0
        lines = content.splitlines()
        for number, line in enumerate(lines):
            if line.strip() == '==':
                count += 1
            if count == instance:
                return self.adjust_line_position(content, number) + 1

class IniParserTest(unittest.TestCase):
    def test_parse(self):
        file_path = os.path.join(os.path.dirname(__file__), 'test_data', 'example.ini')
        parser = IniParser()
        result = parser.parse(file_path)
        self.assertEqual(result, {'settings': '[settings]\na = b\n', 'markup': '###\n<html>\n  <body>\n    <h1>Test</h1>\n  </body>\n</html>'})

    def test_parse_offset(self):
        file_path = os.path.join(os.path.dirname(__file__), 'test_data', 'example.ini')
        parser = IniParser()
        result = parser.parse_offset(file_path)
        self.assertEqual(result, {'settings': 10, 'code': 25, 'markup': 36})

if __name__ == '__main__':
    unittest.main()
[/PYTHON]
