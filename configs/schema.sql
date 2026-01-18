-- Computer Vision Pipeline Database Schema
-- PostgreSQL 15+
--
-- This schema matches the ImageMetadata dataclass from pipeline.py
-- and supports both batch processing and on-demand interactive features.

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Images table (base record for every image)
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 hash for deduplication
    file_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'unknown',  -- places365, user_upload, etc.
    source_category VARCHAR(100),  -- Original dataset category if known
    
    -- Qdrant reference
    embedding_id UUID,  -- Point ID in Qdrant collection
    
    -- Processing tracking
    processing_version VARCHAR(20) DEFAULT '1.0.0',
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_images_file_hash ON images(file_hash);
CREATE INDEX idx_images_source_type ON images(source_type);
CREATE INDEX idx_images_created_at ON images(created_at);

-- ============================================================================
-- LANE A: Always-on metadata (computed for every image)
-- ============================================================================

-- Captions table (VLM-generated descriptions)
CREATE TABLE captions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    
    caption_full TEXT,          -- Full caption from VLM
    caption_short VARCHAR(200), -- Truncated for display
    extracted_nouns TEXT[],     -- Nouns for GroundingDINO prompts
    ocr_text TEXT,              -- OCR-detected text (signs, menus, etc.)
    language_hint VARCHAR(10),  -- Detected language of text in image
    
    model_name VARCHAR(50) DEFAULT 'qwen2.5-vl-3b',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(image_id)  -- One caption per image
);

CREATE INDEX idx_captions_image_id ON captions(image_id);
-- Full-text search on captions
CREATE INDEX idx_captions_full_text ON captions USING GIN (to_tsvector('english', caption_full));

-- Tags table (taxonomy-scored tags)
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    
    tag_id VARCHAR(100) NOT NULL,      -- e.g., 'scene.restaurant.ramen_shop'
    display_name VARCHAR(100) NOT NULL, -- e.g., 'Ramen Shop'
    category VARCHAR(50) NOT NULL,      -- e.g., 'scene', 'mood', 'weather'
    confidence REAL NOT NULL,           -- Cosine similarity score
    rank INTEGER NOT NULL,              -- 1-10 (position in top-K)
    
    source VARCHAR(30) DEFAULT 'siglip_cosine',  -- How tag was assigned
    evidence JSONB,                     -- e.g., {"prompt_used": "ramen shop interior"}
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(image_id, tag_id)
);

CREATE INDEX idx_tags_image_id ON tags(image_id);
CREATE INDEX idx_tags_tag_id ON tags(tag_id);
CREATE INDEX idx_tags_category ON tags(category);
CREATE INDEX idx_tags_confidence ON tags(confidence DESC);
-- Fast lookup for filtering by tag
CREATE INDEX idx_tags_lookup ON tags(tag_id, confidence DESC);

-- ============================================================================
-- LANE B: On-demand interactive metadata (computed when needed)
-- ============================================================================

-- Detections table (GroundingDINO bounding boxes)
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    
    label VARCHAR(100) NOT NULL,        -- Object label
    confidence REAL NOT NULL,           -- Detection confidence
    
    -- Bounding box (normalized 0-1 or pixel coordinates)
    box_x1 REAL NOT NULL,
    box_y1 REAL NOT NULL,
    box_x2 REAL NOT NULL,
    box_y2 REAL NOT NULL,
    box_format VARCHAR(20) DEFAULT 'xyxy_pixels',  -- 'xyxy_pixels' or 'xyxy_normalized'
    
    -- Area for filtering small detections
    area_px INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_detections_image_id ON detections(image_id);
CREATE INDEX idx_detections_label ON detections(label);

-- Masks table (SAM2 segmentation masks)
CREATE TABLE masks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    
    label VARCHAR(100) NOT NULL,
    
    -- RLE-encoded mask (compressed)
    mask_rle TEXT NOT NULL,
    mask_format VARCHAR(20) DEFAULT 'coco_rle',
    
    -- Mask statistics
    area_px INTEGER,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    
    iou_score REAL,  -- SAM2 predicted IoU
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_masks_image_id ON masks(image_id);
CREATE INDEX idx_masks_detection_id ON masks(detection_id);

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- Full image metadata view (for API responses)
CREATE OR REPLACE VIEW v_image_metadata AS
SELECT 
    i.id,
    i.file_hash,
    i.file_path,
    i.width,
    i.height,
    i.source_type,
    i.source_category,
    i.embedding_id,
    i.processed_at,
    c.caption_full,
    c.caption_short,
    c.ocr_text,
    -- Aggregate tags as JSONB array
    (
        SELECT jsonb_agg(jsonb_build_object(
            'tag_id', t.tag_id,
            'display', t.display_name,
            'category', t.category,
            'confidence', t.confidence
        ) ORDER BY t.rank)
        FROM tags t WHERE t.image_id = i.id
    ) AS tags,
    -- Has interactive data?
    EXISTS(SELECT 1 FROM detections d WHERE d.image_id = i.id) AS has_detections,
    EXISTS(SELECT 1 FROM masks m WHERE m.image_id = i.id) AS has_masks
FROM images i
LEFT JOIN captions c ON c.image_id = i.id;

-- Tag statistics view (for taxonomy analysis)
CREATE OR REPLACE VIEW v_tag_stats AS
SELECT 
    tag_id,
    display_name,
    category,
    COUNT(*) as usage_count,
    AVG(confidence) as avg_confidence,
    MAX(confidence) as max_confidence,
    MIN(confidence) as min_confidence
FROM tags
GROUP BY tag_id, display_name, category
ORDER BY usage_count DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER images_updated_at
    BEFORE UPDATE ON images
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Search function (combines text search with tag filtering)
CREATE OR REPLACE FUNCTION search_images(
    query_text TEXT DEFAULT NULL,
    tag_filters TEXT[] DEFAULT NULL,
    source_filter VARCHAR(50) DEFAULT NULL,
    limit_count INTEGER DEFAULT 50
)
RETURNS TABLE (
    image_id UUID,
    file_path TEXT,
    caption_short VARCHAR(200),
    tags JSONB,
    relevance REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.id,
        i.file_path,
        c.caption_short,
        (
            SELECT jsonb_agg(jsonb_build_object('tag_id', t.tag_id, 'display', t.display_name))
            FROM tags t WHERE t.image_id = i.id AND t.rank <= 3
        ),
        CASE 
            WHEN query_text IS NOT NULL THEN
                ts_rank(to_tsvector('english', c.caption_full), plainto_tsquery('english', query_text))
            ELSE 1.0
        END::REAL as relevance
    FROM images i
    LEFT JOIN captions c ON c.image_id = i.id
    WHERE 
        (source_filter IS NULL OR i.source_type = source_filter)
        AND (query_text IS NULL OR to_tsvector('english', c.caption_full) @@ plainto_tsquery('english', query_text))
        AND (tag_filters IS NULL OR EXISTS (
            SELECT 1 FROM tags t 
            WHERE t.image_id = i.id AND t.tag_id = ANY(tag_filters)
        ))
    ORDER BY relevance DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Find images by tag:
-- SELECT * FROM v_image_metadata WHERE tags @> '[{"tag_id": "scene.restaurant"}]';

-- Find images with text in caption:
-- SELECT * FROM v_image_metadata WHERE caption_full ILIKE '%ramen%';

-- Full-text search:
-- SELECT * FROM search_images('cozy restaurant', ARRAY['scene.restaurant'], NULL, 20);

-- Get images needing Lane B processing:
-- SELECT id, file_path FROM images WHERE id NOT IN (SELECT DISTINCT image_id FROM detections);
