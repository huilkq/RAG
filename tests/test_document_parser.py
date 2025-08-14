"""
文档解析器测试

测试各种文档格式的解析功能
"""

import pytest
import tempfile
import os
from pathlib import Path

from app.services.parser_service import (
    DocumentParserService, 
    PDFParser, 
    DocxParser, 
    MarkdownParser, 
    TxtParser
)


class TestDocumentParsers:
    """文档解析器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.parser_service = DocumentParserService()
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_pdf_parser_can_parse(self):
        """测试PDF解析器格式识别"""
        parser = PDFParser()
        
        # 测试支持的格式
        assert parser.can_parse("test.pdf") == True
        assert parser.can_parse("test.PDF") == True
        
        # 测试不支持的格式
        assert parser.can_parse("test.docx") == False
        assert parser.can_parse("test.txt") == False
    
    def test_docx_parser_can_parse(self):
        """测试Word解析器格式识别"""
        parser = DocxParser()
        
        # 测试支持的格式
        assert parser.can_parse("test.docx") == True
        assert parser.can_parse("test.DOCX") == True
        
        # 测试不支持的格式
        assert parser.can_parse("test.pdf") == False
        assert parser.can_parse("test.txt") == False
    
    def test_markdown_parser_can_parse(self):
        """测试Markdown解析器格式识别"""
        parser = MarkdownParser()
        
        # 测试支持的格式
        assert parser.can_parse("test.md") == True
        assert parser.can_parse("test.markdown") == True
        assert parser.can_parse("test.MD") == True
        
        # 测试不支持的格式
        assert parser.can_parse("test.pdf") == False
        assert parser.can_parse("test.txt") == False
    
    def test_txt_parser_can_parse(self):
        """测试文本解析器格式识别"""
        parser = TxtParser()
        
        # 测试支持的格式
        assert parser.can_parse("test.txt") == True
        assert parser.can_parse("test.TXT") == True
        
        # 测试不支持的格式
        assert parser.can_parse("test.pdf") == False
        assert parser.can_parse("test.docx") == False
    
    def test_parser_service_get_parser(self):
        """测试解析器服务获取合适的解析器"""
        # 测试PDF
        pdf_parser = self.parser_service.get_parser("test.pdf")
        assert isinstance(pdf_parser, PDFParser)
        
        # 测试Word
        docx_parser = self.parser_service.get_parser("test.docx")
        assert isinstance(docx_parser, DocxParser)
        
        # 测试Markdown
        md_parser = self.parser_service.get_parser("test.md")
        assert isinstance(md_parser, MarkdownParser)
        
        # 测试文本
        txt_parser = self.parser_service.get_parser("test.txt")
        assert isinstance(txt_parser, TxtParser)
        
        # 测试不支持的格式
        with pytest.raises(ValueError):
            self.parser_service.get_parser("test.unknown")
    
    def test_text_splitting(self):
        """测试文本分段功能"""
        # 创建测试文本
        test_text = "这是一个测试文档。" * 50  # 创建长文本
        
        # 测试分段
        chunks = self.parser_service.create_chunks(
            document_id="test_doc",
            content=test_text,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # 验证分段结果
        assert len(chunks) > 1  # 应该有多个片段
        assert all(len(chunk.content) <= 100 for chunk in chunks)  # 片段长度不超过限制
        
        # 验证片段属性
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "test_doc"
            assert chunk.chunk_index == i
            assert chunk.content.strip()  # 内容不为空
    
    def test_file_validation(self):
        """测试文件验证功能"""
        # 创建测试文件
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("测试内容")
        
        # 测试有效文件
        validation_result = self.parser_service.validate_file(test_file)
        assert validation_result["valid"] == True
        assert len(validation_result["errors"]) == 0
        
        # 测试不存在的文件
        validation_result = self.parser_service.validate_file("nonexistent.txt")
        assert validation_result["valid"] == False
        assert len(validation_result["errors"]) > 0
    
    def test_supported_formats(self):
        """测试支持的文件格式"""
        formats = self.parser_service.get_supported_formats()
        
        expected_formats = ['.pdf', '.docx', '.md', '.markdown', '.txt']
        assert all(fmt in formats for fmt in expected_formats)
    
    def test_parse_and_chunk_integration(self):
        """测试解析和分段的集成功能"""
        # 创建测试文本文件
        test_file = os.path.join(self.test_dir, "test.txt")
        test_content = "这是第一段内容。这是第二段内容。这是第三段内容。" * 20
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # 测试解析和分段
        content, chunks = self.parser_service.parse_and_chunk(
            document_id="test_doc",
            file_path=test_file,
            chunk_size=50,
            chunk_overlap=10
        )
        
        # 验证结果
        assert content == test_content
        assert len(chunks) > 1
        assert all(chunk.document_id == "test_doc" for chunk in chunks)


class TestTextSplitting:
    """文本分段测试类"""
    
    def test_short_text_no_splitting(self):
        """测试短文本不需要分段"""
        parser = PDFParser()  # 使用任意解析器
        
        short_text = "这是一个短文本。"
        chunks = parser.split_text(short_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_long_text_splitting(self):
        """测试长文本分段"""
        parser = PDFParser()
        
        # 创建长文本
        long_text = "这是一个句子。" * 30
        
        chunks = parser.split_text(long_text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_sentence_boundary_splitting(self):
        """测试句子边界分段"""
        parser = PDFParser()
        
        text = "第一句。第二句！第三句？第四句。第五句。"
        
        chunks = parser.split_text(text, chunk_size=20, chunk_overlap=5)
        
        # 验证分段结果
        assert len(chunks) > 1
        # 检查是否在句子边界处分割
        for chunk in chunks:
            if chunk != chunks[-1]:  # 不是最后一段
                assert chunk.endswith(('.', '!', '？', '。')) or chunk.endswith(('。', '！', '?', '.'))
    
    def test_overlap_handling(self):
        """测试重叠处理"""
        parser = PDFParser()
        
        text = "这是一个测试文本。" * 10
        
        chunks = parser.split_text(text, chunk_size=30, chunk_overlap=10)
        
        # 验证重叠
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 检查是否有重叠
            overlap_found = False
            for j in range(min(len(current_chunk), len(next_chunk))):
                if current_chunk[-j:] == next_chunk[:j]:
                    overlap_found = True
                    break
            
            assert overlap_found or len(current_chunk) < 30, f"片段 {i} 和 {i+1} 之间缺少重叠"


if __name__ == "__main__":
    pytest.main([__file__])
