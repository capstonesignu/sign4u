const express = require('express');
const router = express.Router();
const pool = require('../config/db');
const authMiddleware = require('../middlewares/auth');

// GET /api/doctors
router.get('/', async(req, res)=>{
    try{
        const {specialty_id} = req.query;
        
        let query = `
        SELECT d.id, d.name, d.profile_image_url,
        d.experience_years, d.rating,
        s.id as specialty_id, s.name as specialty_name, s.icon_url
        FROM doctors d
        LEFT JOIN specialties s ON d.specialty_id = s.id
        WHERE 1=1
        `;
        const params = [];
        if (specialty_id){
            params.push(specialty_id);
            query+=` AND d.specialty_id=$1`;
        }

        query +=' ORDER BY d.id';
        const result = await pool.query(query, params);

        res.json(result.rows.map(row=>({
            id: row.id, 
            name: row.name,
            profileImageUrl: row.profile_image_url,
            experienceYears: row.experience_years,
            rating: row.rating,
            specialty: row.specialty_id ? {
                id: row.specialty_id,
                name: row.specialty_name, 
                iconUrl: row.icon_url
            } : null
        })));
    }catch(err){
        res.status(500).json({error:err.message});
    }
});

//GET /api/doctors/:id
router.get('/:id', async(req, res)=>{
    try{
        const {id} = req.params;
        const result = await pool.query(`
            SELECT d.id, d.name, d.experience_years, d.rating, d.profile_image_url, s.id as specialty_id, s.name as specialty_name, s.icon_url
            FROM doctors d
            LEFT JOIN specialties s ON d.specialty_id = s.id
            WHERE d.id = $1
            `, [id]);

            if (result.rows.length===0){
                return res.status(404).json({error: '의사를 찾을 수 없습니다.'});
            }
            const row = result.rows[0];
            res.json({
                id: row.id, 
                name: row.name,
                profileImageUrl: row.profile_image_url,
                experienceYears: row.experience_years,
                rating: row.rating,
                specialty: row.specialty_id ? {
                    id: row.specialty_id, 
                    name: row.specialty_name, 
                    iconUrl: row.icon_url
                } : null
            });
    }catch (err){
        res.status(500).json({error: err.message});
    }
});

// PUT /api/doctors/me/specialty — 신규 의사 specialty/hospital 저장
router.put('/me/specialty', authMiddleware, async (req, res) => {
    try {
        const { id, role } = req.user;
        if (role !== 'DOCTOR') {
            return res.status(403).json({ error: '의사만 접근 가능합니다' });
        }

        const { specialtyId, hospitalName } = req.body;
        if (!specialtyId) {
            return res.status(400).json({ error: 'specialtyId는 필수입니다' });
        }

        const spec = await pool.query('SELECT id FROM specialties WHERE id = $1', [specialtyId]);
        if (spec.rows.length === 0) {
            return res.status(404).json({ error: '존재하지 않는 진료과입니다' });
        }

        const result = await pool.query(
            `UPDATE doctors SET specialty_id = $1, hospital_name = $2 WHERE id = $3 RETURNING *`,
            [specialtyId, hospitalName || null, id]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ error: '의사를 찾을 수 없습니다' });
        }

        res.json({ message: '저장 완료', doctor: result.rows[0] });
    } catch (err) {
        console.error('specialty 저장 오류:', err);
        res.status(500).json({ error: '서버 오류가 발생했습니다' });
    }
});

module.exports = router;