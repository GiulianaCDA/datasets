package com.dimtion.shaarlier;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by dimtion on 12/05/2015.
 * Interface between the table TAGS and the JAVA objects
 */
class TagsSource {
    private final String[] allColumns = {MySQLiteHelper.TAGS_COLUMN_ID,
            MySQLiteHelper.TAGS_COLUMN_ID_ACCOUNT,
            MySQLiteHelper.TAGS_COLUMN_TAG};
    private final MySQLiteHelper dbHelper;
    private SQLiteDatabase db;

    public TagsSource(Context context) {
        dbHelper = new MySQLiteHelper(context);
    }

    public void rOpen() throws SQLException {
        db = dbHelper.getReadableDatabase();
    }

    public void wOpen() throws SQLException {
        db = dbHelper.getWritableDatabase();
    }

    public void close() {
        dbHelper.close();
    }

    public List<Tag> getAllTags(ShaarliAccount masterAccount) {
        List<Tag> tags = new ArrayList<>();

        String query = "SELECT * FROM " + MySQLiteHelper.TABLE_TAGS + " WHERE " + MySQLiteHelper.TAGS_COLUMN_ID_ACCOUNT + " = ?";
        Cursor cursor = db.rawQuery(query, new String[] {String.valueOf(masterAccount.getId())});
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            Tag account = cursorToTag(cursor);
            tags.add(account);
            cursor.moveToNext();
        }

        cursor.close();
        return tags;
    }

    public Tag createTag(ShaarliAccount masterAccount, String value) {
        Tag tag = new Tag();
        tag.setMasterAccount(masterAccount);
        tag.setValue(value.trim());

        ContentValues values = new ContentValues();
        values.put(MySQLiteHelper.TAGS_COLUMN_ID_ACCOUNT, masterAccount.getId());
        values.put(MySQLiteHelper.TAGS_COLUMN_TAG, tag.getValue());

        // If existing, do nothing :
        Cursor cursor = db.query(MySQLiteHelper.TABLE_TAGS, allColumns,
                MySQLiteHelper.TAGS_COLUMN_ID_ACCOUNT + " = ? AND " + MySQLiteHelper.TAGS_COLUMN_TAG + " = ?",
                new String[] {String.valueOf(masterAccount.getId()), tag.getValue()}, null, null, null);
        try {
            cursor.moveToFirst();
            if (cursor.isAfterLast()) {
                long insertId = db.insert(MySQLiteHelper.TABLE_TAGS, null, values);
                tag.setId(insertId);
                return tag;
            } else {
                tag = cursorToTag(cursor);
            }
        } catch (Exception e){
            tag = null;
        } finally {
            cursor.close();
        }

        return tag;
    }

    public void deleteAllTags(ShaarliAccount masterAccount) {
        String query = "DELETE FROM " + MySQLiteHelper.TABLE_TAGS + " WHERE " + MySQLiteHelper.TAGS_COLUMN_ID_ACCOUNT + " = ?";
        db.execSQL(query, new Object[] {String.valueOf(masterAccount.getId())});
    }

    private Tag cursorToTag(Cursor cursor) {
        Tag tag = new Tag();
        tag.setId(cursor.getInt(0));
        tag.setMasterAccount(new ShaarliAccount(cursor.getInt(1)));
        tag.setValue(cursor.getString(2));
        return tag;
    }
}
