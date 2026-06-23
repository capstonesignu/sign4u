const express = require('express');
const router = express.Router();
const pool = require('../config/db');

// GET api/specialties
router.get('/', async(req, res)=>{
    try{
        const result = await pool.query(
            'SELECT id, name, icon_url as "iconUrl" FROM specialties ORDER BY id'
        );
        res.json(result.rows);
    } catch(err){
        res.status(500).json({error:err.message});
    }
});

module.exports = router;