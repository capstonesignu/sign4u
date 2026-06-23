const express = require('express');
const router = express.Router();
const pool=require('../config/db');
const auth = require('../middlewares/auth');

// GET /api/users/me
router.get('/me', auth, async(req,  res)=>{
    try{
        const userId= req.user.id;
        const role = req.user.role;
        let result;
        if (role === 'DOCTOR') {
          result = await pool.query(`
            SELECT d.id, d.email, d.name, d.profile_image_url, d.hospital_name, d.experience_years, d.created_at,
                   s.id as specialty_id, s.name as specialty_name, s.icon_url
            FROM doctors d
            LEFT JOIN specialties s ON d.specialty_id = s.id
            WHERE d.id = $1
          `, [userId]);
        } else {
          result = await pool.query(`
            SELECT id, email, name, profile_image_url, created_at
            FROM patients WHERE id = $1
          `, [userId]);
        }

            if(result.rows.length===0){
                return res.status(404).json({error: '사용자를 찾을 수 없습니다'});
            }
            const user = result.rows[0];
            res.json({
                id: user.id,
                email: user.email,
                name: user.name,
                role,
                profileImageUrl: user.profile_image_url,
                hospital: user.hospital_name || null,
                experienceYears: user.experience_years || null,
                specialty: user.specialty_id ? {
                    id: user.specialty_id,
                    name: user.specialty_name,
                    iconUrl: user.icon_url
                } : null,
                createdAt: user.created_at
            });     
    }catch(err){
        res.status(500).json({error:err.message});
    }
});

// PATCH /api/users/me
router.patch('/me', auth, async(req, res)=>{
    try{
        const userId = req.user.id;
        const {name} = req.body;

        const role = req.user.role;
        let result;
        if (role === 'DOCTOR') {
          result = await pool.query(`
            UPDATE doctors SET name = $1, updated_at = NOW()
            WHERE id = $2 RETURNING id, name, profile_image_url, updated_at
          `, [name, userId]);
        } else {
          result = await pool.query(`
            UPDATE patients SET name = $1, updated_at = NOW()
            WHERE id = $2 RETURNING id, name, profile_image_url, updated_at
          `, [name, userId]);
        }

            const user = result.rows[0];
            res.json({
                id: user.id,
                name: user.name,
                profileImageUrl: user.profile_image_url,
                updatedAt: user.updated_at
            });
    } catch (err){
        res.status(500).json({error:err.message});
    }
});



module.exports = router;