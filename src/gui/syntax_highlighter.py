import re
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt6.QtCore import Qt

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)

        self.highlightingRules = []

        # Keywords
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(QColor("#c678dd")) # Purple
        keywordFormat.setFontWeight(QFont.Weight.Bold)
        keywords = [
            '\\bdef\\b', '\\bclass\\b', '\\bimport\\b', '\\bfrom\\b',
            '\\bif\\b', '\\belif\\b', '\\belse\\b', '\\bfor\\b',
            '\\bwhile\\b', '\\bbreak\\b', '\\bcontinue\\b', '\\breturn\\b',
            '\\bpass\\b', '\\byield\\b', '\\bNone\\b', '\\bTrue\\b', '\\bFalse\\b',
            '\\basync\\b', '\\bawait\\b', '\\band\\b', '\\bor\\b', '\\bnot\\b',
            '\\bis\\b', '\\bin\\b', '\\btry\\b', '\\bexcept\\b', '\\bfinally\\b',
            '\\bwith\\b', '\\bas\\b', '\\bassert\\b', '\\bdel\\b', '\\bglobal\\b',
            '\\bnonlocal\\b', '\\blambda\\b'
        ]
        for pattern in keywords:
            self.highlightingRules.append((re.compile(pattern), keywordFormat))

        # Builtins
        builtinFormat = QTextCharFormat()
        builtinFormat.setForeground(QColor("#56b6c2")) # Cyan
        builtins = [
            '\\bprint\\b', '\\blen\\b', '\\brange\\b', '\\bstr\\b',
            '\\bint\\b', '\\bfloat\\b', '\\bdict\\b', '\\bset\\b', '\\btype\\b',
            '\\bdir\\b', '\\bgetattr\\b', '\\bsetattr\\b', '\\bhasattr\\b'
        ]
        for pattern in builtins:
            self.highlightingRules.append((re.compile(pattern), builtinFormat))

        # Decorators
        decoratorFormat = QTextCharFormat()
        decoratorFormat.setForeground(QColor("#61afef")) # Blue
        self.highlightingRules.append((re.compile(r'@[^\n]*'), decoratorFormat))

        # Strings (single line)
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(QColor("#98c379")) # Green
        self.highlightingRules.append((re.compile(r'".*"'), stringFormat))
        self.highlightingRules.append((re.compile(r"'.*'"), stringFormat))

        # Functions
        functionFormat = QTextCharFormat()
        functionFormat.setForeground(QColor("#61afef")) # Blue
        self.highlightingRules.append((re.compile(r'\b[A-Za-z0-9_]+(?=\()'), functionFormat))

        # Numbers
        numberFormat = QTextCharFormat()
        numberFormat.setForeground(QColor("#d19a66")) # Orange
        self.highlightingRules.append((re.compile(r'\b[0-9]+(\.[0-9]+)?\b'), numberFormat))

        # Single Line Comments
        self.commentFormat = QTextCharFormat()
        self.commentFormat.setForeground(QColor("#7f848e")) # Gray
        self.commentFormat.setFontItalic(True)
        self.highlightingRules.append((re.compile(r'#[^\n]*'), self.commentFormat))

        # Multiline strings/comments
        self.multiLineStringFormat = QTextCharFormat()
        self.multiLineStringFormat.setForeground(QColor("#98c379"))
        self.multiLineStringFormat.setFontItalic(True)

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)

        # Basic Multiline String highlighting logic (simplistic)
        self.setCurrentBlockState(0)
        
        in_string = False
        start_idx = 0
        if self.previousBlockState() == 1:
            in_string = True
            
        # Very crude """ detection for walkthrough purposes
        idx = 0
        while idx < len(text):
            if text[idx:idx+3] == '"""' or text[idx:idx+3] == "'''":
                if in_string:
                    self.setFormat(start_idx, idx + 3 - start_idx, self.multiLineStringFormat)
                    in_string = False
                else:
                    in_string = True
                    start_idx = idx
                idx += 3
                continue
            idx += 1
            
        if in_string:
            self.setCurrentBlockState(1)
            self.setFormat(start_idx, len(text) - start_idx, self.multiLineStringFormat)

class JsonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)

        self.highlightingRules = []

        # Keys (strings before colon)
        keyFormat = QTextCharFormat()
        keyFormat.setForeground(QColor("#61afef")) # Blue
        self.highlightingRules.append((re.compile(r'"[^"]*"\s*:'), keyFormat))

        # String values
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(QColor("#98c379")) # Green
        self.highlightingRules.append((re.compile(r':\s*"[^"]*"'), stringFormat))

        # Numbers
        numberFormat = QTextCharFormat()
        numberFormat.setForeground(QColor("#d19a66")) # Orange
        self.highlightingRules.append((re.compile(r'\b[0-9]+(\.[0-9]+)?\b'), numberFormat))

        # Booleans / Null
        boolFormat = QTextCharFormat()
        boolFormat.setForeground(QColor("#c678dd")) # Purple
        self.highlightingRules.append((re.compile(r'\b(true|false|null)\b'), boolFormat))
        
    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            for match in pattern.finditer(text):
                # For keys, only highlight the string part, not the colon
                if pattern.pattern == r'"[^"]*"\s*:':
                    # Find where the quote ends to skip the colon
                    s = match.group()
                    idx = s.rfind('"') + 1
                    self.setFormat(match.start(), idx, format)
                # For string values, only highlight the string part, not the colon
                elif pattern.pattern == r':\s*"[^"]*"':
                    s = match.group()
                    idx = s.find('"')
                    self.setFormat(match.start() + idx, len(s) - idx, format)
                else:
                    self.setFormat(match.start(), match.end() - match.start(), format)
